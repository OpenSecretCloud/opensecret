//! Bounded, transport-v2-only persistence for the platform control plane.
//!
//! Transport v1 deliberately keeps using its existing handlers and model
//! loaders. This module gives transport v2 a narrower boundary: every stored
//! response is measured before database-controlled strings or JSON are loaded,
//! list reads have an independent row cap, and mutations revalidate the live
//! platform actor and organization role inside the committing transaction.

use chrono::{DateTime, Duration, Utc};
use diesel::dsl::{count_star, sql};
use diesel::prelude::*;
use diesel::sql_types::{BigInt, Integer};
use diesel::{Connection, OptionalExtension};
use serde_json::Value;
use uuid::Uuid;

use crate::models::org_memberships::OrgRole;
use crate::models::project_settings::{EmailSettings, OAuthSettings, SettingCategory};
use crate::models::schema::{
    invite_codes, org_memberships, org_project_secrets, org_projects, orgs,
    platform_email_verifications, platform_users, project_settings,
};
use crate::web::platform::common::{
    DetailedInviteResponse, InviteResponse, MeResponse, MembershipResponse, OrgResponse,
    PlatformUserResponse, ProjectResponse, SecretResponse,
};

type Pool = diesel::r2d2::Pool<diesel::r2d2::ConnectionManager<PgConnection>>;

/// Platform lists are intentionally unpaginated for compatibility. This cap
/// prevents a database attacker from turning that contract into an unbounded
/// metadata allocation even when all stored strings are tiny.
pub(crate) const MAX_PLATFORM_RESOURCE_ROWS: usize = 65_536;

const ACCOUNTED_BYTES_PER_LIST_ROW: usize = 256;
// The shared v1 validators bound these fields to 50 Unicode scalar values,
// not 50 bytes. Four bytes per scalar is therefore the exact UTF-8 ceiling
// for values that existing clients can legitimately store.
const MAX_ORG_NAME_BYTES: usize = 50 * 4;
const MAX_PROJECT_NAME_BYTES: usize = 50 * 4;
const MAX_PROJECT_DESCRIPTION_BYTES: usize = 255 * 4;
const MAX_PLATFORM_NAME_BYTES: usize = 50 * 4;
const MAX_EMAIL_BYTES: usize = 255 * 4;
const MAX_SECRET_KEY_BYTES: usize = 50;
const MAX_SETTINGS_JSON_BYTES: usize = 64 * 1024;
const MEMBERSHIP_ROLE_RANK_SQL: &str = "CASE org_memberships.role WHEN 'owner' THEN 0 WHEN 'admin' THEN 1 WHEN 'developer' THEN 2 WHEN 'viewer' THEN 3 ELSE -1 END";
const INVITE_ROLE_RANK_SQL: &str = "CASE invite_codes.role WHEN 'owner' THEN 0 WHEN 'admin' THEN 1 WHEN 'developer' THEN 2 WHEN 'viewer' THEN 3 ELSE -1 END";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PlatformResourceKind {
    Actor,
    Organization,
    Project,
    Secret,
    EmailSettings,
    Membership,
    Invite,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum PlatformResourceError {
    #[error("{0:?} not found")]
    NotFound(PlatformResourceKind),
    #[error("the bound platform actor is not authorized for this operation")]
    Unauthorized,
    #[error("the request violates the platform resource contract")]
    Validation,
    #[error("the requested platform resource conflicts with existing state")]
    Conflict,
    #[error("the last organization owner cannot be removed or demoted")]
    LastOwner,
    #[error("the invite has already been used")]
    InviteAlreadyUsed,
    #[error("the invite has expired")]
    InviteExpired,
    #[error("an owner invitation requires a verified recipient email")]
    VerifiedEmailRequired,
    #[error("stored platform output exceeds the logical response limit")]
    OutputTooLarge,
    #[error("stored platform state violates its bounded representation")]
    InconsistentSnapshot,
    #[error("database connection unavailable")]
    Connection,
    #[error("database error: {0}")]
    Database(#[from] diesel::result::Error),
}

pub(crate) struct CreatedInvite {
    pub(crate) response: InviteResponse,
    pub(crate) dispatch: InviteEmailDispatch,
}

pub(crate) struct InviteEmailDispatch {
    pub(crate) email: String,
    pub(crate) organization_name: String,
    pub(crate) organization_id: Uuid,
    pub(crate) invite_code: Uuid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StrictRole {
    Owner,
    Admin,
    Developer,
    Viewer,
}

impl StrictRole {
    fn from_rank(rank: i32) -> Result<Self, PlatformResourceError> {
        match rank {
            0 => Ok(Self::Owner),
            1 => Ok(Self::Admin),
            2 => Ok(Self::Developer),
            3 => Ok(Self::Viewer),
            _ => Err(PlatformResourceError::InconsistentSnapshot),
        }
    }

    fn from_org_role(role: &OrgRole) -> Self {
        match role {
            OrgRole::Owner => Self::Owner,
            OrgRole::Admin => Self::Admin,
            OrgRole::Developer => Self::Developer,
            OrgRole::Viewer => Self::Viewer,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Owner => "owner",
            Self::Admin => "admin",
            Self::Developer => "developer",
            Self::Viewer => "viewer",
        }
    }

    fn can_read(self) -> bool {
        matches!(
            self,
            Self::Owner | Self::Admin | Self::Developer | Self::Viewer
        )
    }

    fn can_administer(self) -> bool {
        matches!(self, Self::Owner | Self::Admin)
    }

    fn is_owner(self) -> bool {
        self == Self::Owner
    }
}

fn checked_length(length: i64) -> Result<usize, PlatformResourceError> {
    usize::try_from(length).map_err(|_| PlatformResourceError::InconsistentSnapshot)
}

fn checked_optional_length(length: Option<i64>) -> Result<Option<usize>, PlatformResourceError> {
    length.map(checked_length).transpose()
}

fn account_bytes(
    total: &mut usize,
    bytes: usize,
    logical_body_limit: usize,
) -> Result<(), PlatformResourceError> {
    *total = total
        .checked_add(bytes)
        .ok_or(PlatformResourceError::OutputTooLarge)?;
    if *total > logical_body_limit {
        return Err(PlatformResourceError::OutputTooLarge);
    }
    Ok(())
}

fn validate_list_aggregate(
    row_count: i64,
    aggregate_string_bytes: Option<i64>,
    maximum_field_bytes: Option<i64>,
    field_limit: usize,
    logical_body_limit: usize,
) -> Result<(usize, usize), PlatformResourceError> {
    let rows = checked_length(row_count)?;
    if rows > MAX_PLATFORM_RESOURCE_ROWS {
        return Err(PlatformResourceError::OutputTooLarge);
    }
    let aggregate = checked_optional_length(aggregate_string_bytes)?.unwrap_or(0);
    let maximum = checked_optional_length(maximum_field_bytes)?.unwrap_or(0);
    if maximum > field_limit {
        return Err(PlatformResourceError::InconsistentSnapshot);
    }
    let mut total = rows
        .checked_mul(ACCOUNTED_BYTES_PER_LIST_ROW)
        .ok_or(PlatformResourceError::OutputTooLarge)?;
    account_bytes(&mut total, aggregate, logical_body_limit)?;
    Ok((rows, aggregate))
}

fn validate_single_output(
    lengths: impl IntoIterator<Item = (i64, usize)>,
    logical_body_limit: usize,
) -> Result<Vec<usize>, PlatformResourceError> {
    let mut total = ACCOUNTED_BYTES_PER_LIST_ROW;
    let mut expected = Vec::new();
    for (length, maximum) in lengths {
        let length = checked_length(length)?;
        if length > maximum {
            return Err(PlatformResourceError::InconsistentSnapshot);
        }
        account_bytes(&mut total, length, logical_body_limit)?;
        expected.push(length);
    }
    Ok(expected)
}

fn get_connection(
    pool: &Pool,
) -> Result<
    diesel::r2d2::PooledConnection<diesel::r2d2::ConnectionManager<PgConnection>>,
    PlatformResourceError,
> {
    pool.get().map_err(|_| PlatformResourceError::Connection)
}

fn ensure_live_actor(
    conn: &mut PgConnection,
    actor_id: Uuid,
) -> Result<i32, PlatformResourceError> {
    platform_users::table
        .filter(platform_users::uuid.eq(actor_id))
        .select(platform_users::id)
        .first::<i32>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(PlatformResourceKind::Actor))
}

fn lock_live_actor(conn: &mut PgConnection, actor_id: Uuid) -> Result<i32, PlatformResourceError> {
    platform_users::table
        .filter(platform_users::uuid.eq(actor_id))
        .select(platform_users::id)
        .for_update()
        .first::<i32>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(PlatformResourceKind::Actor))
}

fn find_org(conn: &mut PgConnection, org_uuid: Uuid) -> Result<i32, PlatformResourceError> {
    orgs::table
        .filter(orgs::uuid.eq(org_uuid))
        .select(orgs::id)
        .first::<i32>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(
            PlatformResourceKind::Organization,
        ))
}

fn lock_org(conn: &mut PgConnection, org_uuid: Uuid) -> Result<i32, PlatformResourceError> {
    orgs::table
        .filter(orgs::uuid.eq(org_uuid))
        .select(orgs::id)
        .for_update()
        .first::<i32>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(
            PlatformResourceKind::Organization,
        ))
}

fn decode_role_flags(
    owner: bool,
    admin: bool,
    developer: bool,
    viewer: bool,
) -> Result<StrictRole, PlatformResourceError> {
    match (owner, admin, developer, viewer) {
        (true, false, false, false) => Ok(StrictRole::Owner),
        (false, true, false, false) => Ok(StrictRole::Admin),
        (false, false, true, false) => Ok(StrictRole::Developer),
        (false, false, false, true) => Ok(StrictRole::Viewer),
        _ => Err(PlatformResourceError::InconsistentSnapshot),
    }
}

fn membership_role(
    conn: &mut PgConnection,
    actor_id: Uuid,
    org_id: i32,
) -> Result<StrictRole, PlatformResourceError> {
    let flags = org_memberships::table
        .filter(org_memberships::platform_user_id.eq(actor_id))
        .filter(org_memberships::org_id.eq(org_id))
        .select((
            org_memberships::role.eq(StrictRole::Owner.as_str()),
            org_memberships::role.eq(StrictRole::Admin.as_str()),
            org_memberships::role.eq(StrictRole::Developer.as_str()),
            org_memberships::role.eq(StrictRole::Viewer.as_str()),
        ))
        .first::<(bool, bool, bool, bool)>(conn)
        .optional()?
        .ok_or(PlatformResourceError::Unauthorized)?;
    decode_role_flags(flags.0, flags.1, flags.2, flags.3)
}

fn locked_membership_role(
    conn: &mut PgConnection,
    actor_id: Uuid,
    org_id: i32,
) -> Result<StrictRole, PlatformResourceError> {
    let flags = org_memberships::table
        .filter(org_memberships::platform_user_id.eq(actor_id))
        .filter(org_memberships::org_id.eq(org_id))
        .select((
            org_memberships::role.eq(StrictRole::Owner.as_str()),
            org_memberships::role.eq(StrictRole::Admin.as_str()),
            org_memberships::role.eq(StrictRole::Developer.as_str()),
            org_memberships::role.eq(StrictRole::Viewer.as_str()),
        ))
        .for_update()
        .first::<(bool, bool, bool, bool)>(conn)
        .optional()?
        .ok_or(PlatformResourceError::Unauthorized)?;
    decode_role_flags(flags.0, flags.1, flags.2, flags.3)
}

fn require_read(role: StrictRole) -> Result<(), PlatformResourceError> {
    if role.can_read() {
        Ok(())
    } else {
        Err(PlatformResourceError::Unauthorized)
    }
}

fn require_admin(role: StrictRole) -> Result<(), PlatformResourceError> {
    if role.can_administer() {
        Ok(())
    } else {
        Err(PlatformResourceError::Unauthorized)
    }
}

fn require_owner(role: StrictRole) -> Result<(), PlatformResourceError> {
    if role.is_owner() {
        Ok(())
    } else {
        Err(PlatformResourceError::Unauthorized)
    }
}

fn find_project(
    conn: &mut PgConnection,
    org_id: i32,
    project_uuid: Uuid,
) -> Result<i32, PlatformResourceError> {
    org_projects::table
        .filter(org_projects::org_id.eq(org_id))
        .filter(org_projects::uuid.eq(project_uuid))
        .select(org_projects::id)
        .first::<i32>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(
            PlatformResourceKind::Project,
        ))
}

fn lock_project(
    conn: &mut PgConnection,
    org_id: i32,
    project_uuid: Uuid,
) -> Result<i32, PlatformResourceError> {
    org_projects::table
        .filter(org_projects::org_id.eq(org_id))
        .filter(org_projects::uuid.eq(project_uuid))
        .select(org_projects::id)
        .for_update()
        .first::<i32>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(
            PlatformResourceKind::Project,
        ))
}

fn map_unique_conflict(error: diesel::result::Error) -> PlatformResourceError {
    if matches!(
        error,
        diesel::result::Error::DatabaseError(diesel::result::DatabaseErrorKind::UniqueViolation, _)
    ) {
        PlatformResourceError::Conflict
    } else {
        PlatformResourceError::Database(error)
    }
}

fn load_orgs_for_actor(
    conn: &mut PgConnection,
    actor_id: Uuid,
    logical_body_limit: usize,
) -> Result<Vec<OrgResponse>, PlatformResourceError> {
    let (count, total_name_bytes, max_name_bytes, invalid_roles) = org_memberships::table
        .inner_join(orgs::table.on(orgs::id.eq(org_memberships::org_id)))
        .filter(org_memberships::platform_user_id.eq(actor_id))
        .select((
            count_star(),
            sql::<diesel::sql_types::Nullable<BigInt>>(
                "SUM(octet_length(orgs.name)::bigint)::bigint",
            ),
            sql::<diesel::sql_types::Nullable<BigInt>>(
                "MAX(octet_length(orgs.name)::bigint)::bigint",
            ),
            sql::<diesel::sql_types::Nullable<BigInt>>(
                "SUM(CASE WHEN org_memberships.role IN ('owner','admin','developer','viewer') THEN 0 ELSE 1 END)::bigint",
            ),
        ))
        .first::<(i64, Option<i64>, Option<i64>, Option<i64>)>(conn)?;
    if invalid_roles.unwrap_or(0) != 0 {
        return Err(PlatformResourceError::InconsistentSnapshot);
    }
    let (expected_rows, expected_name_bytes) = validate_list_aggregate(
        count,
        total_name_bytes,
        max_name_bytes,
        MAX_ORG_NAME_BYTES,
        logical_body_limit,
    )?;

    let rows = org_memberships::table
        .inner_join(orgs::table.on(orgs::id.eq(org_memberships::org_id)))
        .filter(org_memberships::platform_user_id.eq(actor_id))
        .order(orgs::id.asc())
        .select((orgs::uuid, orgs::name))
        .load::<(Uuid, String)>(conn)?;
    let actual_name_bytes = rows.iter().try_fold(0_usize, |total, (_, name)| {
        if name.len() > MAX_ORG_NAME_BYTES {
            return Err(PlatformResourceError::InconsistentSnapshot);
        }
        total
            .checked_add(name.len())
            .ok_or(PlatformResourceError::InconsistentSnapshot)
    })?;
    if rows.len() != expected_rows || actual_name_bytes != expected_name_bytes {
        return Err(PlatformResourceError::InconsistentSnapshot);
    }
    Ok(rows
        .into_iter()
        .map(|(id, name)| OrgResponse { id, name })
        .collect())
}

pub(crate) async fn get_me(
    pool: &Pool,
    actor_id: Uuid,
    logical_body_limit: usize,
) -> Result<MeResponse, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            let measurement = platform_users::table
                .filter(platform_users::uuid.eq(actor_id))
                .select((
                    platform_users::id,
                    sql::<BigInt>("octet_length(platform_users.email::text)::bigint"),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "octet_length(platform_users.name)::bigint",
                    ),
                    platform_users::created_at,
                    platform_users::updated_at,
                ))
                .first::<(i32, i64, Option<i64>, DateTime<Utc>, DateTime<Utc>)>(conn)
                .optional()?
                .ok_or(PlatformResourceError::NotFound(PlatformResourceKind::Actor))?;

            let mut lengths = vec![(measurement.1, MAX_EMAIL_BYTES)];
            if let Some(name_length) = measurement.2 {
                lengths.push((name_length, MAX_PLATFORM_NAME_BYTES));
            }
            let expected = validate_single_output(lengths, logical_body_limit)?;
            let (email, name) = platform_users::table
                .filter(platform_users::id.eq(measurement.0))
                .select((platform_users::email, platform_users::name))
                .first::<(String, Option<String>)>(conn)?;
            if email.len() != expected[0]
                || name.as_ref().map(String::len) != expected.get(1).copied()
            {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }

            let email_verified = platform_email_verifications::table
                .filter(platform_email_verifications::platform_user_id.eq(actor_id))
                .filter(platform_email_verifications::is_verified.eq(true))
                .select(platform_email_verifications::id)
                .first::<i32>(conn)
                .optional()?
                .is_some();
            let actor_bytes = ACCOUNTED_BYTES_PER_LIST_ROW
                .checked_add(email.len())
                .and_then(|total| total.checked_add(name.as_ref().map_or(0, String::len)))
                .ok_or(PlatformResourceError::OutputTooLarge)?;
            let remaining_body_limit = logical_body_limit
                .checked_sub(actor_bytes)
                .ok_or(PlatformResourceError::OutputTooLarge)?;
            let organizations = load_orgs_for_actor(conn, actor_id, remaining_body_limit)?;

            Ok(MeResponse {
                user: PlatformUserResponse {
                    id: actor_id,
                    email,
                    name,
                    email_verified,
                    created_at: measurement.3,
                    updated_at: measurement.4,
                },
                organizations,
            })
        })
}

pub(crate) async fn create_org(
    pool: &Pool,
    actor_id: Uuid,
    name: String,
) -> Result<OrgResponse, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org = diesel::insert_into(orgs::table)
            .values(orgs::name.eq(&name))
            .returning((orgs::id, orgs::uuid))
            .get_result::<(i32, Uuid)>(conn)
            .map_err(map_unique_conflict)?;
        diesel::insert_into(org_memberships::table)
            .values((
                org_memberships::platform_user_id.eq(actor_id),
                org_memberships::org_id.eq(org.0),
                org_memberships::role.eq(StrictRole::Owner.as_str()),
            ))
            .execute(conn)
            .map_err(map_unique_conflict)?;
        Ok(OrgResponse { id: org.1, name })
    })
}

pub(crate) async fn list_orgs(
    pool: &Pool,
    actor_id: Uuid,
    logical_body_limit: usize,
) -> Result<Vec<OrgResponse>, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            ensure_live_actor(conn, actor_id)?;
            load_orgs_for_actor(conn, actor_id, logical_body_limit)
        })
}

pub(crate) async fn delete_org(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
) -> Result<(), PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        require_owner(locked_membership_role(conn, actor_id, org_id)?)?;
        let deleted = diesel::delete(orgs::table.filter(orgs::id.eq(org_id))).execute(conn)?;
        if deleted == 1 {
            Ok(())
        } else {
            Err(PlatformResourceError::NotFound(
                PlatformResourceKind::Organization,
            ))
        }
    })
}

pub(crate) async fn create_project(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    name: String,
    description: Option<String>,
) -> Result<ProjectResponse, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        require_admin(locked_membership_role(conn, actor_id, org_id)?)?;
        let description = Some(description.unwrap_or_default());
        let row = diesel::insert_into(org_projects::table)
            .values((
                org_projects::org_id.eq(org_id),
                org_projects::name.eq(&name),
                org_projects::description.eq(&description),
                org_projects::status.eq("active"),
            ))
            .returning((
                org_projects::uuid,
                org_projects::client_id,
                org_projects::created_at,
            ))
            .get_result::<(Uuid, Uuid, DateTime<Utc>)>(conn)
            .map_err(map_unique_conflict)?;
        Ok(ProjectResponse {
            id: row.0,
            client_id: row.1,
            name,
            description,
            status: "active".to_string(),
            created_at: row.2,
        })
    })
}

fn status_from_rank(rank: i32) -> Result<&'static str, PlatformResourceError> {
    match rank {
        0 => Ok("active"),
        1 => Ok("inactive"),
        2 => Ok("suspended"),
        _ => Err(PlatformResourceError::InconsistentSnapshot),
    }
}

fn load_project_response(
    conn: &mut PgConnection,
    project_id: i32,
    logical_body_limit: usize,
) -> Result<ProjectResponse, PlatformResourceError> {
    let measurement = org_projects::table
        .filter(org_projects::id.eq(project_id))
        .select((
            sql::<BigInt>("octet_length(org_projects.name)::bigint"),
            sql::<diesel::sql_types::Nullable<BigInt>>(
                "octet_length(org_projects.description)::bigint",
            ),
            sql::<Integer>(
                "CASE org_projects.status WHEN 'active' THEN 0 WHEN 'inactive' THEN 1 WHEN 'suspended' THEN 2 ELSE -1 END",
            ),
        ))
        .first::<(i64, Option<i64>, i32)>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(
            PlatformResourceKind::Project,
        ))?;
    let mut lengths = vec![(measurement.0, MAX_PROJECT_NAME_BYTES)];
    if let Some(description_length) = measurement.1 {
        lengths.push((description_length, MAX_PROJECT_DESCRIPTION_BYTES));
    }
    let expected = validate_single_output(lengths, logical_body_limit)?;
    let row = org_projects::table
        .filter(org_projects::id.eq(project_id))
        .select((
            org_projects::uuid,
            org_projects::client_id,
            org_projects::name,
            org_projects::description,
            sql::<Integer>(
                "CASE org_projects.status WHEN 'active' THEN 0 WHEN 'inactive' THEN 1 WHEN 'suspended' THEN 2 ELSE -1 END",
            ),
            org_projects::created_at,
        ))
        .first::<(Uuid, Uuid, String, Option<String>, i32, DateTime<Utc>)>(conn)?;
    if row.2.len() != expected[0] || row.3.as_ref().map(String::len) != expected.get(1).copied() {
        return Err(PlatformResourceError::InconsistentSnapshot);
    }
    Ok(ProjectResponse {
        id: row.0,
        client_id: row.1,
        name: row.2,
        description: row.3,
        status: status_from_rank(row.4)?.to_string(),
        created_at: row.5,
    })
}

pub(crate) async fn list_projects(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<Vec<ProjectResponse>, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            ensure_live_actor(conn, actor_id)?;
            let org_id = find_org(conn, org_uuid)?;
            require_read(membership_role(conn, actor_id, org_id)?)?;
            let (count, total, maximum_name, maximum_description, invalid_statuses) =
                org_projects::table
                .filter(org_projects::org_id.eq(org_id))
                .select((
                    count_star(),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "SUM(octet_length(name)::bigint + COALESCE(octet_length(description)::bigint, 0))::bigint",
                    ),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "MAX(octet_length(name)::bigint)::bigint",
                    ),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "MAX(COALESCE(octet_length(description)::bigint, 0))::bigint",
                    ),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "SUM(CASE WHEN status IN ('active','inactive','suspended') THEN 0 ELSE 1 END)::bigint",
                    ),
                ))
                .first::<(i64, Option<i64>, Option<i64>, Option<i64>, Option<i64>)>(conn)?;
            if invalid_statuses.unwrap_or(0) != 0 {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            if checked_optional_length(maximum_description)?.unwrap_or(0)
                > MAX_PROJECT_DESCRIPTION_BYTES
            {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            let (expected_rows, expected_bytes) = validate_list_aggregate(
                count,
                total,
                maximum_name,
                MAX_PROJECT_NAME_BYTES,
                logical_body_limit,
            )?;
            let rows = org_projects::table
                .filter(org_projects::org_id.eq(org_id))
                .order(org_projects::id.asc())
                .select((
                    org_projects::uuid,
                    org_projects::client_id,
                    org_projects::name,
                    org_projects::description,
                    sql::<Integer>(
                        "CASE status WHEN 'active' THEN 0 WHEN 'inactive' THEN 1 WHEN 'suspended' THEN 2 ELSE -1 END",
                    ),
                    org_projects::created_at,
                ))
                .load::<(Uuid, Uuid, String, Option<String>, i32, DateTime<Utc>)>(conn)?;
            let actual_bytes = rows.iter().try_fold(0_usize, |total, row| {
                if row.2.len() > MAX_PROJECT_NAME_BYTES
                    || row.3.as_ref().is_some_and(|value| value.len() > MAX_PROJECT_DESCRIPTION_BYTES)
                {
                    return Err(PlatformResourceError::InconsistentSnapshot);
                }
                status_from_rank(row.4)?;
                total
                    .checked_add(row.2.len())
                    .and_then(|total| total.checked_add(row.3.as_ref().map_or(0, String::len)))
                    .ok_or(PlatformResourceError::InconsistentSnapshot)
            })?;
            if rows.len() != expected_rows || actual_bytes != expected_bytes {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            rows.into_iter()
                .map(|row| {
                    Ok(ProjectResponse {
                        id: row.0,
                        client_id: row.1,
                        name: row.2,
                        description: row.3,
                        status: status_from_rank(row.4)?.to_string(),
                        created_at: row.5,
                    })
                })
                .collect()
        })
}

pub(crate) async fn get_project(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<ProjectResponse, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            ensure_live_actor(conn, actor_id)?;
            let org_id = find_org(conn, org_uuid)?;
            require_read(membership_role(conn, actor_id, org_id)?)?;
            let project_id = find_project(conn, org_id, project_uuid)?;
            load_project_response(conn, project_id, logical_body_limit)
        })
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn update_project(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
    name: Option<String>,
    description: Option<String>,
    status: Option<String>,
    logical_body_limit: usize,
) -> Result<ProjectResponse, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        require_admin(locked_membership_role(conn, actor_id, org_id)?)?;
        let project_id = lock_project(conn, org_id, project_uuid)?;
        let current = load_project_response(conn, project_id, logical_body_limit)?;
        let final_name = name.unwrap_or(current.name);
        let final_description = description.or(current.description);
        let final_status = status.unwrap_or(current.status);
        let updated = diesel::update(org_projects::table.filter(org_projects::id.eq(project_id)))
            .set((
                org_projects::name.eq(&final_name),
                org_projects::description.eq(&final_description),
                org_projects::status.eq(&final_status),
                org_projects::updated_at.eq(diesel::dsl::now),
            ))
            .execute(conn)
            .map_err(map_unique_conflict)?;
        if updated != 1 {
            return Err(PlatformResourceError::NotFound(
                PlatformResourceKind::Project,
            ));
        }
        Ok(ProjectResponse {
            id: current.id,
            client_id: current.client_id,
            name: final_name,
            description: final_description,
            status: final_status,
            created_at: current.created_at,
        })
    })
}

pub(crate) async fn delete_project(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
) -> Result<(), PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        require_admin(locked_membership_role(conn, actor_id, org_id)?)?;
        let project_id = lock_project(conn, org_id, project_uuid)?;
        let deleted = diesel::delete(org_projects::table.filter(org_projects::id.eq(project_id)))
            .execute(conn)?;
        if deleted == 1 {
            Ok(())
        } else {
            Err(PlatformResourceError::NotFound(
                PlatformResourceKind::Project,
            ))
        }
    })
}

pub(crate) async fn create_secret(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
    key_name: String,
    encrypted_secret: &[u8],
) -> Result<SecretResponse, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        require_admin(locked_membership_role(conn, actor_id, org_id)?)?;
        let project_id = lock_project(conn, org_id, project_uuid)?;
        let timestamps = diesel::insert_into(org_project_secrets::table)
            .values((
                org_project_secrets::project_id.eq(project_id),
                org_project_secrets::key_name.eq(&key_name),
                org_project_secrets::secret_enc.eq(encrypted_secret),
            ))
            .on_conflict((
                org_project_secrets::project_id,
                org_project_secrets::key_name,
            ))
            .do_update()
            .set(org_project_secrets::secret_enc.eq(encrypted_secret))
            .returning((
                org_project_secrets::created_at,
                org_project_secrets::updated_at,
            ))
            .get_result::<(DateTime<Utc>, DateTime<Utc>)>(conn)?;
        Ok(SecretResponse {
            key_name,
            created_at: timestamps.0,
            updated_at: timestamps.1,
        })
    })
}

pub(crate) async fn list_secrets(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<Vec<SecretResponse>, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            ensure_live_actor(conn, actor_id)?;
            let org_id = find_org(conn, org_uuid)?;
            require_read(membership_role(conn, actor_id, org_id)?)?;
            let project_id = find_project(conn, org_id, project_uuid)?;
            let (count, total, maximum) = org_project_secrets::table
                .filter(org_project_secrets::project_id.eq(project_id))
                .select((
                    count_star(),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "SUM(octet_length(key_name)::bigint)::bigint",
                    ),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "MAX(octet_length(key_name)::bigint)::bigint",
                    ),
                ))
                .first::<(i64, Option<i64>, Option<i64>)>(conn)?;
            let (expected_rows, expected_bytes) = validate_list_aggregate(
                count,
                total,
                maximum,
                MAX_SECRET_KEY_BYTES,
                logical_body_limit,
            )?;
            let rows = org_project_secrets::table
                .filter(org_project_secrets::project_id.eq(project_id))
                .order(org_project_secrets::id.asc())
                .select((
                    org_project_secrets::key_name,
                    org_project_secrets::created_at,
                    org_project_secrets::updated_at,
                ))
                .load::<(String, DateTime<Utc>, DateTime<Utc>)>(conn)?;
            let actual_bytes = rows.iter().try_fold(0_usize, |total, row| {
                if row.0.len() > MAX_SECRET_KEY_BYTES {
                    return Err(PlatformResourceError::InconsistentSnapshot);
                }
                total
                    .checked_add(row.0.len())
                    .ok_or(PlatformResourceError::InconsistentSnapshot)
            })?;
            if rows.len() != expected_rows || actual_bytes != expected_bytes {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            Ok(rows
                .into_iter()
                .map(|row| SecretResponse {
                    key_name: row.0,
                    created_at: row.1,
                    updated_at: row.2,
                })
                .collect())
        })
}

pub(crate) async fn delete_secret(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
    key_name: &str,
) -> Result<(), PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        require_admin(locked_membership_role(conn, actor_id, org_id)?)?;
        let project_id = lock_project(conn, org_id, project_uuid)?;
        let deleted = diesel::delete(
            org_project_secrets::table
                .filter(org_project_secrets::project_id.eq(project_id))
                .filter(org_project_secrets::key_name.eq(key_name)),
        )
        .execute(conn)?;
        if deleted == 1 {
            Ok(())
        } else {
            Err(PlatformResourceError::NotFound(
                PlatformResourceKind::Secret,
            ))
        }
    })
}

fn load_settings_json(
    conn: &mut PgConnection,
    project_id: i32,
    category: SettingCategory,
    logical_body_limit: usize,
) -> Result<Option<Value>, PlatformResourceError> {
    let category_name = category.as_str();
    let measured = project_settings::table
        .filter(project_settings::project_id.eq(project_id))
        .filter(project_settings::category.eq(category_name))
        .select(sql::<BigInt>(
            "octet_length(project_settings.settings::text)::bigint",
        ))
        .first::<i64>(conn)
        .optional()?;
    let Some(measured) = measured else {
        return Ok(None);
    };
    let expected =
        validate_single_output([(measured, MAX_SETTINGS_JSON_BYTES)], logical_body_limit)?[0];
    let value = project_settings::table
        .filter(project_settings::project_id.eq(project_id))
        .filter(project_settings::category.eq(category_name))
        .select(project_settings::settings)
        .first::<Value>(conn)?;
    let encoded =
        serde_json::to_vec(&value).map_err(|_| PlatformResourceError::InconsistentSnapshot)?;
    if encoded.len() > expected || encoded.len() > MAX_SETTINGS_JSON_BYTES {
        return Err(PlatformResourceError::InconsistentSnapshot);
    }
    Ok(Some(value))
}

fn serialize_bounded_settings<T: serde::Serialize>(
    settings: &T,
) -> Result<Value, PlatformResourceError> {
    let value = serde_json::to_value(settings).map_err(|_| PlatformResourceError::Validation)?;
    let encoded = serde_json::to_vec(&value).map_err(|_| PlatformResourceError::Validation)?;
    if encoded.len() > MAX_SETTINGS_JSON_BYTES {
        return Err(PlatformResourceError::Validation);
    }
    Ok(value)
}

fn authorize_project_read(
    conn: &mut PgConnection,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
) -> Result<i32, PlatformResourceError> {
    ensure_live_actor(conn, actor_id)?;
    let org_id = find_org(conn, org_uuid)?;
    require_read(membership_role(conn, actor_id, org_id)?)?;
    find_project(conn, org_id, project_uuid)
}

fn authorize_project_write(
    conn: &mut PgConnection,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
) -> Result<i32, PlatformResourceError> {
    lock_live_actor(conn, actor_id)?;
    let org_id = lock_org(conn, org_uuid)?;
    require_admin(locked_membership_role(conn, actor_id, org_id)?)?;
    lock_project(conn, org_id, project_uuid)
}

pub(crate) async fn get_email_settings(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<EmailSettings, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            let project_id = authorize_project_read(conn, actor_id, org_uuid, project_uuid)?;
            let value =
                load_settings_json(conn, project_id, SettingCategory::Email, logical_body_limit)?
                    .ok_or(PlatformResourceError::NotFound(
                    PlatformResourceKind::EmailSettings,
                ))?;
            serde_json::from_value(value).map_err(|_| PlatformResourceError::InconsistentSnapshot)
        })
}

pub(crate) async fn update_email_settings(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
    settings: EmailSettings,
) -> Result<EmailSettings, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        let project_id = authorize_project_write(conn, actor_id, org_uuid, project_uuid)?;
        let value = serialize_bounded_settings(&settings)?;
        diesel::insert_into(project_settings::table)
            .values((
                project_settings::project_id.eq(project_id),
                project_settings::category.eq(SettingCategory::Email.as_str()),
                project_settings::settings.eq(value),
            ))
            .on_conflict((project_settings::project_id, project_settings::category))
            .do_update()
            .set(
                project_settings::settings.eq(diesel::upsert::excluded(project_settings::settings)),
            )
            .execute(conn)?;
        Ok(settings)
    })
}

pub(crate) async fn get_oauth_settings(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<OAuthSettings, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            let project_id = authorize_project_read(conn, actor_id, org_uuid, project_uuid)?;
            let Some(value) =
                load_settings_json(conn, project_id, SettingCategory::OAuth, logical_body_limit)?
            else {
                return Ok(OAuthSettings::default());
            };
            serde_json::from_value(value).map_err(|_| PlatformResourceError::InconsistentSnapshot)
        })
}

pub(crate) async fn update_oauth_settings(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    project_uuid: Uuid,
    settings: OAuthSettings,
) -> Result<OAuthSettings, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        let project_id = authorize_project_write(conn, actor_id, org_uuid, project_uuid)?;
        let value = serialize_bounded_settings(&settings)?;
        diesel::insert_into(project_settings::table)
            .values((
                project_settings::project_id.eq(project_id),
                project_settings::category.eq(SettingCategory::OAuth.as_str()),
                project_settings::settings.eq(value),
            ))
            .on_conflict((project_settings::project_id, project_settings::category))
            .do_update()
            .set(
                project_settings::settings.eq(diesel::upsert::excluded(project_settings::settings)),
            )
            .execute(conn)?;
        Ok(settings)
    })
}

pub(crate) async fn list_memberships(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<Vec<MembershipResponse>, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            ensure_live_actor(conn, actor_id)?;
            let org_id = find_org(conn, org_uuid)?;
            require_read(membership_role(conn, actor_id, org_id)?)?;
            let (count, total, maximum, invalid_roles) = org_memberships::table
                .inner_join(
                    platform_users::table
                        .on(platform_users::uuid.eq(org_memberships::platform_user_id)),
                )
                .filter(org_memberships::org_id.eq(org_id))
                .select((
                    count_star(),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "SUM(COALESCE(octet_length(platform_users.name)::bigint, 0))::bigint",
                    ),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "MAX(COALESCE(octet_length(platform_users.name)::bigint, 0))::bigint",
                    ),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "SUM(CASE WHEN org_memberships.role IN ('owner','admin','developer','viewer') THEN 0 ELSE 1 END)::bigint",
                    ),
                ))
                .first::<(i64, Option<i64>, Option<i64>, Option<i64>)>(conn)?;
            if invalid_roles.unwrap_or(0) != 0 {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            let (expected_rows, expected_bytes) = validate_list_aggregate(
                count,
                total,
                maximum,
                MAX_PLATFORM_NAME_BYTES,
                logical_body_limit,
            )?;
            let rows = org_memberships::table
                .inner_join(
                    platform_users::table
                        .on(platform_users::uuid.eq(org_memberships::platform_user_id)),
                )
                .filter(org_memberships::org_id.eq(org_id))
                .order(org_memberships::id.asc())
                .select((
                    org_memberships::platform_user_id,
                    sql::<Integer>(MEMBERSHIP_ROLE_RANK_SQL),
                    platform_users::name,
                ))
                .load::<(Uuid, i32, Option<String>)>(conn)?;
            let actual_bytes = rows.iter().try_fold(0_usize, |total, row| {
                StrictRole::from_rank(row.1)?;
                if row.2.as_ref().is_some_and(|name| name.len() > MAX_PLATFORM_NAME_BYTES) {
                    return Err(PlatformResourceError::InconsistentSnapshot);
                }
                total
                    .checked_add(row.2.as_ref().map_or(0, String::len))
                    .ok_or(PlatformResourceError::InconsistentSnapshot)
            })?;
            if rows.len() != expected_rows || actual_bytes != expected_bytes {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            rows.into_iter()
                .map(|row| {
                    Ok(MembershipResponse {
                        user_id: row.0,
                        role: StrictRole::from_rank(row.1)?.as_str().to_string(),
                        name: row.2,
                    })
                })
                .collect()
        })
}

#[derive(QueryableByName)]
struct OwnerCount {
    #[diesel(sql_type = BigInt)]
    count: i64,
}

fn lock_and_count_owners(
    conn: &mut PgConnection,
    org_id: i32,
) -> Result<i64, PlatformResourceError> {
    diesel::sql_query(
        "SELECT COUNT(*)::bigint AS count FROM (\
         SELECT id FROM org_memberships \
         WHERE org_id = $1 AND role = 'owner' FOR UPDATE\
         ) AS locked_owners",
    )
    .bind::<Integer, _>(org_id)
    .get_result::<OwnerCount>(conn)
    .map(|row| row.count)
    .map_err(PlatformResourceError::Database)
}

fn lock_target_membership(
    conn: &mut PgConnection,
    org_id: i32,
    user_id: Uuid,
) -> Result<(i32, StrictRole), PlatformResourceError> {
    let row = org_memberships::table
        .filter(org_memberships::org_id.eq(org_id))
        .filter(org_memberships::platform_user_id.eq(user_id))
        .select((
            org_memberships::id,
            org_memberships::role.eq(StrictRole::Owner.as_str()),
            org_memberships::role.eq(StrictRole::Admin.as_str()),
            org_memberships::role.eq(StrictRole::Developer.as_str()),
            org_memberships::role.eq(StrictRole::Viewer.as_str()),
        ))
        .for_update()
        .first::<(i32, bool, bool, bool, bool)>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(
            PlatformResourceKind::Membership,
        ))?;
    Ok((row.0, decode_role_flags(row.1, row.2, row.3, row.4)?))
}

fn load_bounded_platform_name(
    conn: &mut PgConnection,
    user_id: Uuid,
    logical_body_limit: usize,
) -> Result<Option<String>, PlatformResourceError> {
    let length = platform_users::table
        .filter(platform_users::uuid.eq(user_id))
        .select(sql::<diesel::sql_types::Nullable<BigInt>>(
            "octet_length(platform_users.name)::bigint",
        ))
        .first::<Option<i64>>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(
            PlatformResourceKind::Membership,
        ))?;
    let expected = if let Some(length) = length {
        Some(validate_single_output([(length, MAX_PLATFORM_NAME_BYTES)], logical_body_limit)?[0])
    } else {
        None
    };
    let expected_database_length = expected
        .map(i64::try_from)
        .transpose()
        .map_err(|_| PlatformResourceError::InconsistentSnapshot)?;
    let name = platform_users::table
        .filter(platform_users::uuid.eq(user_id))
        // Do not materialize a concurrently enlarged value. The nullable
        // length predicate keeps the second statement safe under READ
        // COMMITTED without adding an arbitrary target-user row lock to the
        // organization mutation lock order.
        .filter(
            sql::<diesel::sql_types::Nullable<BigInt>>("octet_length(platform_users.name)::bigint")
                .is_not_distinct_from(expected_database_length),
        )
        .select(platform_users::name)
        .first::<Option<String>>(conn)
        .optional()?
        .ok_or(PlatformResourceError::InconsistentSnapshot)?;
    if name.as_ref().map(String::len) != expected {
        return Err(PlatformResourceError::InconsistentSnapshot);
    }
    Ok(name)
}

pub(crate) async fn update_membership(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    target_user_id: Uuid,
    new_role: OrgRole,
    logical_body_limit: usize,
) -> Result<MembershipResponse, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        require_owner(locked_membership_role(conn, actor_id, org_id)?)?;
        let (membership_id, old_role) = lock_target_membership(conn, org_id, target_user_id)?;
        let new_role = StrictRole::from_org_role(&new_role);
        if old_role.is_owner() && !new_role.is_owner() && lock_and_count_owners(conn, org_id)? <= 1
        {
            return Err(PlatformResourceError::LastOwner);
        }
        let updated =
            diesel::update(org_memberships::table.filter(org_memberships::id.eq(membership_id)))
                .set((
                    org_memberships::role.eq(new_role.as_str()),
                    org_memberships::updated_at.eq(diesel::dsl::now),
                ))
                .execute(conn)?;
        if updated != 1 {
            return Err(PlatformResourceError::NotFound(
                PlatformResourceKind::Membership,
            ));
        }
        let name = load_bounded_platform_name(conn, target_user_id, logical_body_limit)?;
        Ok(MembershipResponse {
            user_id: target_user_id,
            role: new_role.as_str().to_string(),
            name,
        })
    })
}

pub(crate) async fn delete_membership(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    target_user_id: Uuid,
) -> Result<(), PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        require_owner(locked_membership_role(conn, actor_id, org_id)?)?;
        let (membership_id, target_role) = lock_target_membership(conn, org_id, target_user_id)?;
        if target_role.is_owner() && lock_and_count_owners(conn, org_id)? <= 1 {
            return Err(PlatformResourceError::LastOwner);
        }
        let deleted =
            diesel::delete(org_memberships::table.filter(org_memberships::id.eq(membership_id)))
                .execute(conn)?;
        if deleted == 1 {
            Ok(())
        } else {
            Err(PlatformResourceError::NotFound(
                PlatformResourceKind::Membership,
            ))
        }
    })
}

fn load_bounded_org_name(
    conn: &mut PgConnection,
    org_id: i32,
    logical_body_limit: usize,
) -> Result<(Uuid, String), PlatformResourceError> {
    let measurement = orgs::table
        .filter(orgs::id.eq(org_id))
        .select((orgs::uuid, sql::<BigInt>("octet_length(orgs.name)::bigint")))
        .first::<(Uuid, i64)>(conn)
        .optional()?
        .ok_or(PlatformResourceError::NotFound(
            PlatformResourceKind::Organization,
        ))?;
    let expected =
        validate_single_output([(measurement.1, MAX_ORG_NAME_BYTES)], logical_body_limit)?[0];
    let name = orgs::table
        .filter(orgs::id.eq(org_id))
        .select(orgs::name)
        .first::<String>(conn)?;
    if name.len() != expected {
        return Err(PlatformResourceError::InconsistentSnapshot);
    }
    Ok((measurement.0, name))
}

pub(crate) async fn create_invite(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    email: String,
    invite_role: OrgRole,
    logical_body_limit: usize,
) -> Result<CreatedInvite, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        let actor_role = locked_membership_role(conn, actor_id, org_id)?;
        require_admin(actor_role)?;
        let invite_role = StrictRole::from_org_role(&invite_role);
        if invite_role.is_owner() && !actor_role.is_owner() {
            return Err(PlatformResourceError::Unauthorized);
        }
        let (organization_id, organization_name) =
            load_bounded_org_name(conn, org_id, logical_body_limit)?;
        let code = Uuid::new_v4();
        let expires_at = Utc::now() + Duration::hours(24);
        let row = diesel::insert_into(invite_codes::table)
            .values((
                invite_codes::code.eq(code),
                invite_codes::org_id.eq(org_id),
                invite_codes::email.eq(&email),
                invite_codes::role.eq(invite_role.as_str()),
                invite_codes::expires_at.eq(expires_at),
            ))
            .returning((
                invite_codes::used,
                invite_codes::created_at,
                invite_codes::updated_at,
            ))
            .get_result::<(bool, DateTime<Utc>, DateTime<Utc>)>(conn)
            .map_err(map_unique_conflict)?;
        Ok(CreatedInvite {
            response: InviteResponse {
                code,
                email: email.clone(),
                role: invite_role.as_str().to_string(),
                used: row.0,
                expires_at,
                created_at: row.1,
                updated_at: row.2,
            },
            dispatch: InviteEmailDispatch {
                email,
                organization_name,
                organization_id,
                invite_code: code,
            },
        })
    })
}

type InviteRow = (
    Uuid,
    String,
    i32,
    bool,
    DateTime<Utc>,
    DateTime<Utc>,
    DateTime<Utc>,
);

fn invite_response(row: InviteRow) -> Result<InviteResponse, PlatformResourceError> {
    Ok(InviteResponse {
        code: row.0,
        email: row.1,
        role: StrictRole::from_rank(row.2)?.as_str().to_string(),
        used: row.3,
        expires_at: row.4,
        created_at: row.5,
        updated_at: row.6,
    })
}

pub(crate) async fn list_invites(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    logical_body_limit: usize,
) -> Result<Vec<InviteResponse>, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            ensure_live_actor(conn, actor_id)?;
            let org_id = find_org(conn, org_uuid)?;
            require_admin(membership_role(conn, actor_id, org_id)?)?;
            let now = Utc::now();
            let (count, total, maximum, invalid_roles) = invite_codes::table
                .filter(invite_codes::org_id.eq(org_id))
                .filter(invite_codes::used.eq(false))
                .filter(invite_codes::expires_at.gt(now))
                .select((
                    count_star(),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "SUM(octet_length(email)::bigint)::bigint",
                    ),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "MAX(octet_length(email)::bigint)::bigint",
                    ),
                    sql::<diesel::sql_types::Nullable<BigInt>>(
                        "SUM(CASE WHEN role IN ('owner','admin','developer','viewer') THEN 0 ELSE 1 END)::bigint",
                    ),
                ))
                .first::<(i64, Option<i64>, Option<i64>, Option<i64>)>(conn)?;
            if invalid_roles.unwrap_or(0) != 0 {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            let (expected_rows, expected_bytes) = validate_list_aggregate(
                count,
                total,
                maximum,
                MAX_EMAIL_BYTES,
                logical_body_limit,
            )?;
            let rows = invite_codes::table
                .filter(invite_codes::org_id.eq(org_id))
                .filter(invite_codes::used.eq(false))
                .filter(invite_codes::expires_at.gt(now))
                .order(invite_codes::id.asc())
                .select((
                    invite_codes::code,
                    invite_codes::email,
                    sql::<Integer>(INVITE_ROLE_RANK_SQL),
                    invite_codes::used,
                    invite_codes::expires_at,
                    invite_codes::created_at,
                    invite_codes::updated_at,
                ))
                .load::<InviteRow>(conn)?;
            let actual_bytes = rows.iter().try_fold(0_usize, |total, row| {
                StrictRole::from_rank(row.2)?;
                if row.1.len() > MAX_EMAIL_BYTES {
                    return Err(PlatformResourceError::InconsistentSnapshot);
                }
                total
                    .checked_add(row.1.len())
                    .ok_or(PlatformResourceError::InconsistentSnapshot)
            })?;
            if rows.len() != expected_rows || actual_bytes != expected_bytes {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            rows.into_iter().map(invite_response).collect()
        })
}

pub(crate) async fn get_invite(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    invite_code: Uuid,
    logical_body_limit: usize,
) -> Result<DetailedInviteResponse, PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.build_transaction()
        .read_only()
        .repeatable_read()
        .run::<_, PlatformResourceError, _>(|conn| {
            ensure_live_actor(conn, actor_id)?;
            let org_id = find_org(conn, org_uuid)?;
            let org = load_bounded_org_name(conn, org_id, logical_body_limit)?;
            let measurement = invite_codes::table
                .filter(invite_codes::org_id.eq(org_id))
                .filter(invite_codes::code.eq(invite_code))
                .select((
                    invite_codes::id,
                    sql::<BigInt>("octet_length(invite_codes.email)::bigint"),
                    sql::<Integer>(INVITE_ROLE_RANK_SQL),
                ))
                .first::<(i32, i64, i32)>(conn)
                .optional()?
                .ok_or(PlatformResourceError::NotFound(
                    PlatformResourceKind::Invite,
                ))?;
            let expected_email =
                validate_single_output([(measurement.1, MAX_EMAIL_BYTES)], logical_body_limit)?[0];
            StrictRole::from_rank(measurement.2)?;
            let actor_email_length = platform_users::table
                .filter(platform_users::uuid.eq(actor_id))
                .select(sql::<BigInt>(
                    "octet_length(platform_users.email::text)::bigint",
                ))
                .first::<i64>(conn)?;
            let expected_actor_email = validate_single_output(
                [(actor_email_length, MAX_EMAIL_BYTES)],
                logical_body_limit,
            )?[0];
            let actor_email = platform_users::table
                .filter(platform_users::uuid.eq(actor_id))
                .select(platform_users::email)
                .first::<String>(conn)?;
            if actor_email.len() != expected_actor_email {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            let row = invite_codes::table
                .filter(invite_codes::id.eq(measurement.0))
                .select((
                    invite_codes::code,
                    invite_codes::email,
                    sql::<Integer>(INVITE_ROLE_RANK_SQL),
                    invite_codes::used,
                    invite_codes::expires_at,
                    invite_codes::created_at,
                    invite_codes::updated_at,
                ))
                .first::<InviteRow>(conn)?;
            if row.1.len() != expected_email || row.2 != measurement.2 {
                return Err(PlatformResourceError::InconsistentSnapshot);
            }
            let is_admin = match membership_role(conn, actor_id, org_id) {
                Ok(role) => role.can_administer(),
                Err(PlatformResourceError::Unauthorized) => false,
                Err(error) => return Err(error),
            };
            if actor_email != row.1 && !is_admin {
                return Err(PlatformResourceError::Unauthorized);
            }
            Ok(DetailedInviteResponse {
                code: row.0,
                email: row.1,
                role: StrictRole::from_rank(row.2)?.as_str().to_string(),
                used: row.3,
                expires_at: row.4,
                created_at: row.5,
                updated_at: row.6,
                organization_name: org.1,
            })
        })
}

pub(crate) async fn delete_invite(
    pool: &Pool,
    actor_id: Uuid,
    org_uuid: Uuid,
    invite_code: Uuid,
) -> Result<(), PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let org_id = lock_org(conn, org_uuid)?;
        require_admin(locked_membership_role(conn, actor_id, org_id)?)?;
        let deleted = diesel::delete(
            invite_codes::table
                .filter(invite_codes::org_id.eq(org_id))
                .filter(invite_codes::code.eq(invite_code)),
        )
        .execute(conn)?;
        if deleted == 1 {
            Ok(())
        } else {
            Err(PlatformResourceError::NotFound(
                PlatformResourceKind::Invite,
            ))
        }
    })
}

pub(crate) async fn accept_invite(
    pool: &Pool,
    actor_id: Uuid,
    invite_code: Uuid,
    logical_body_limit: usize,
) -> Result<(), PlatformResourceError> {
    let mut conn = get_connection(pool)?;
    conn.transaction::<_, PlatformResourceError, _>(|conn| {
        lock_live_actor(conn, actor_id)?;
        let actor_measurement = platform_users::table
            .filter(platform_users::uuid.eq(actor_id))
            .select(sql::<BigInt>(
                "octet_length(platform_users.email::text)::bigint",
            ))
            .first::<i64>(conn)?;
        let expected_actor_email =
            validate_single_output([(actor_measurement, MAX_EMAIL_BYTES)], logical_body_limit)?[0];
        let actor_email = platform_users::table
            .filter(platform_users::uuid.eq(actor_id))
            .select(platform_users::email)
            .first::<String>(conn)?;
        if actor_email.len() != expected_actor_email {
            return Err(PlatformResourceError::InconsistentSnapshot);
        }

        // Preserve the mutation lock order used by organization-scoped
        // operations: actor -> organization -> child row. Reading the fixed
        // internal ID first is safe; the locked invite is re-scoped to that
        // same organization before any authorization state is consumed.
        let invite_org_id = invite_codes::table
            .filter(invite_codes::code.eq(invite_code))
            .select(invite_codes::org_id)
            .first::<i32>(conn)
            .optional()?
            .ok_or(PlatformResourceError::NotFound(
                PlatformResourceKind::Invite,
            ))?;
        orgs::table
            .filter(orgs::id.eq(invite_org_id))
            .select(orgs::id)
            .for_update()
            .first::<i32>(conn)
            .optional()?
            .ok_or(PlatformResourceError::NotFound(
                PlatformResourceKind::Invite,
            ))?;
        let measurement = invite_codes::table
            .filter(invite_codes::code.eq(invite_code))
            .filter(invite_codes::org_id.eq(invite_org_id))
            .select((
                invite_codes::id,
                invite_codes::org_id,
                sql::<BigInt>("octet_length(invite_codes.email)::bigint"),
                sql::<Integer>(INVITE_ROLE_RANK_SQL),
                invite_codes::used,
                invite_codes::expires_at,
            ))
            .for_update()
            .first::<(i32, i32, i64, i32, bool, DateTime<Utc>)>(conn)
            .optional()?
            .ok_or(PlatformResourceError::NotFound(
                PlatformResourceKind::Invite,
            ))?;
        let expected_invite_email =
            validate_single_output([(measurement.2, MAX_EMAIL_BYTES)], logical_body_limit)?[0];
        let invite_role = StrictRole::from_rank(measurement.3)?;
        if measurement.4 {
            return Err(PlatformResourceError::InviteAlreadyUsed);
        }
        if measurement.5 < Utc::now() {
            return Err(PlatformResourceError::InviteExpired);
        }
        let invite_email = invite_codes::table
            .filter(invite_codes::id.eq(measurement.0))
            .select(invite_codes::email)
            .first::<String>(conn)?;
        if invite_email.len() != expected_invite_email {
            return Err(PlatformResourceError::InconsistentSnapshot);
        }
        if invite_email != actor_email {
            return Err(PlatformResourceError::Unauthorized);
        }
        if invite_role.is_owner() {
            let verified = platform_email_verifications::table
                .filter(platform_email_verifications::platform_user_id.eq(actor_id))
                .filter(platform_email_verifications::is_verified.eq(true))
                .select(platform_email_verifications::id)
                .first::<i32>(conn)
                .optional()?
                .is_some();
            if !verified {
                return Err(PlatformResourceError::VerifiedEmailRequired);
            }
        }

        diesel::insert_into(org_memberships::table)
            .values((
                org_memberships::platform_user_id.eq(actor_id),
                org_memberships::org_id.eq(measurement.1),
                org_memberships::role.eq(invite_role.as_str()),
            ))
            .execute(conn)
            .map_err(map_unique_conflict)?;
        let updated = diesel::update(
            invite_codes::table
                .filter(invite_codes::id.eq(measurement.0))
                .filter(invite_codes::used.eq(false)),
        )
        .set((
            invite_codes::used.eq(true),
            invite_codes::updated_at.eq(diesel::dsl::now),
        ))
        .execute(conn)?;
        if updated != 1 {
            return Err(PlatformResourceError::InviteAlreadyUsed);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::{
        accept_invite, create_invite, create_org, create_project, list_invites, list_memberships,
        list_secrets, update_membership, validate_list_aggregate, PlatformResourceError, Pool,
        StrictRole, MAX_ORG_NAME_BYTES, MAX_PLATFORM_RESOURCE_ROWS, MAX_PROJECT_NAME_BYTES,
    };
    use crate::models::org_memberships::OrgRole;
    use crate::models::platform_users::NewPlatformUser;
    use crate::models::schema::{
        invite_codes, org_memberships, org_project_secrets, org_projects, orgs,
        platform_email_verifications, platform_users,
    };
    use crate::web::platform::common::CreateOrgRequest;
    use chrono::{Duration, Utc};
    use diesel::prelude::*;
    use diesel::r2d2::ConnectionManager;
    use uuid::Uuid;
    use validator::Validate;

    struct TestActor {
        id: Uuid,
        email: String,
    }

    struct PlatformFixture {
        pool: Pool,
        actors: Vec<Uuid>,
        organizations: Vec<Uuid>,
    }

    impl PlatformFixture {
        fn new(pool: Pool) -> Self {
            Self {
                pool,
                actors: Vec::new(),
                organizations: Vec::new(),
            }
        }

        fn actor(&mut self, label: &str) -> TestActor {
            let marker = Uuid::new_v4();
            let email = format!("platform-v2-{label}-{marker}@example.com");
            let mut conn = self.pool.get().expect("test database connection");
            let user = NewPlatformUser::new(email.clone(), None)
                .with_name(format!("{label}-{marker}"))
                .insert(&mut conn)
                .expect("test platform user should insert");
            self.actors.push(user.uuid);
            TestActor {
                id: user.uuid,
                email,
            }
        }

        fn track_org(&mut self, org_id: Uuid) {
            self.organizations.push(org_id);
        }

        fn org_internal_id(&self, org_uuid: Uuid) -> i32 {
            let mut conn = self.pool.get().expect("test database connection");
            orgs::table
                .filter(orgs::uuid.eq(org_uuid))
                .select(orgs::id)
                .first(&mut conn)
                .expect("test organization should exist")
        }

        fn add_member(&self, org_uuid: Uuid, actor_id: Uuid, role: StrictRole) {
            let org_id = self.org_internal_id(org_uuid);
            let mut conn = self.pool.get().expect("test database connection");
            diesel::insert_into(org_memberships::table)
                .values((
                    org_memberships::platform_user_id.eq(actor_id),
                    org_memberships::org_id.eq(org_id),
                    org_memberships::role.eq(role.as_str()),
                ))
                .execute(&mut conn)
                .expect("test membership should insert");
        }
    }

    impl Drop for PlatformFixture {
        fn drop(&mut self) {
            let Ok(mut conn) = self.pool.get() else {
                return;
            };
            let _ = diesel::delete(
                platform_email_verifications::table
                    .filter(platform_email_verifications::platform_user_id.eq_any(&self.actors)),
            )
            .execute(&mut conn);
            let _ = diesel::delete(orgs::table.filter(orgs::uuid.eq_any(&self.organizations)))
                .execute(&mut conn);
            let _ = diesel::delete(
                platform_users::table.filter(platform_users::uuid.eq_any(&self.actors)),
            )
            .execute(&mut conn);
        }
    }

    fn disposable_pool() -> Option<Pool> {
        let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
            eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
            return None;
        };
        let manager = ConnectionManager::<PgConnection>::new(database_url);
        Some(
            diesel::r2d2::Pool::builder()
                .max_size(6)
                .build(manager)
                .expect("connect to disposable migrated PostgreSQL"),
        )
    }

    #[test]
    fn aggregate_bounds_reject_row_field_and_logical_overflow() {
        assert_eq!(
            validate_list_aggregate(2, Some(10), Some(7), 8, 1_024)
                .expect("small aggregate should fit"),
            (2, 10)
        );
        assert!(matches!(
            validate_list_aggregate(
                i64::try_from(MAX_PLATFORM_RESOURCE_ROWS + 1).unwrap(),
                Some(0),
                Some(0),
                8,
                usize::MAX,
            ),
            Err(PlatformResourceError::OutputTooLarge)
        ));
        assert!(matches!(
            validate_list_aggregate(1, Some(9), Some(9), 8, 1_024),
            Err(PlatformResourceError::InconsistentSnapshot)
        ));
        assert!(matches!(
            validate_list_aggregate(4, Some(1), Some(1), 8, 1_024),
            Err(PlatformResourceError::OutputTooLarge)
        ));
    }

    #[test]
    fn name_bounds_cover_unicode_whitespace_accepted_by_shared_validation() {
        let name = format!("A{}B", "\u{2003}".repeat(48));
        assert_eq!(name.chars().count(), 50);
        assert!(CreateOrgRequest { name: name.clone() }.validate().is_ok());
        assert!(name.len() > 50);
        assert!(name.len() <= MAX_ORG_NAME_BYTES);
        assert!(name.len() <= MAX_PROJECT_NAME_BYTES);
    }

    #[tokio::test]
    #[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
    async fn database_secret_metadata_is_bounded_owner_scoped_and_ciphertext_free() {
        let Some(pool) = disposable_pool() else {
            return;
        };
        let mut fixture = PlatformFixture::new(pool.clone());
        let owner = fixture.actor("secret-owner");
        let outsider = fixture.actor("secret-outsider");
        let org = create_org(&pool, owner.id, "Secret Metadata Org".to_string())
            .await
            .expect("owner should create organization");
        fixture.track_org(org.id);
        let project = create_project(
            &pool,
            owner.id,
            org.id,
            "Secret Metadata Project".to_string(),
            None,
        )
        .await
        .expect("owner should create project");

        let mut conn = pool.get().expect("test database connection");
        let project_id = org_projects::table
            .filter(org_projects::uuid.eq(project.id))
            .select(org_projects::id)
            .first::<i32>(&mut conn)
            .expect("test project should exist");
        diesel::insert_into(org_project_secrets::table)
            .values((
                org_project_secrets::project_id.eq(project_id),
                org_project_secrets::key_name.eq("SAFE_KEY"),
                // A metadata list with a tiny response budget must not load or
                // account this large ciphertext column.
                org_project_secrets::secret_enc.eq(vec![7_u8; 2 * 1024 * 1024]),
            ))
            .execute(&mut conn)
            .expect("test secret should insert");
        drop(conn);

        let listed = list_secrets(&pool, owner.id, org.id, project.id, 1_024)
            .await
            .expect("bounded metadata should not touch ciphertext");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].key_name, "SAFE_KEY");
        assert!(matches!(
            list_secrets(&pool, outsider.id, org.id, project.id, 1_024).await,
            Err(PlatformResourceError::Unauthorized)
        ));

        let mut conn = pool.get().expect("test database connection");
        diesel::update(
            org_project_secrets::table
                .filter(org_project_secrets::project_id.eq(project_id))
                .filter(org_project_secrets::key_name.eq("SAFE_KEY")),
        )
        .set(org_project_secrets::key_name.eq("X".repeat(51)))
        .execute(&mut conn)
        .expect("test database tamper should update key name");
        drop(conn);
        assert!(matches!(
            list_secrets(&pool, owner.id, org.id, project.id, 1_024).await,
            Err(PlatformResourceError::InconsistentSnapshot)
        ));
    }

    #[tokio::test]
    #[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
    async fn database_role_checks_and_last_owner_transition_are_transactional() {
        let Some(pool) = disposable_pool() else {
            return;
        };
        let mut fixture = PlatformFixture::new(pool.clone());
        let owner = fixture.actor("rbac-owner");
        let viewer = fixture.actor("rbac-viewer");
        let org = create_org(&pool, owner.id, "RBAC Org".to_string())
            .await
            .expect("owner should create organization");
        fixture.track_org(org.id);
        fixture.add_member(org.id, viewer.id, StrictRole::Viewer);

        let mut conn = pool.get().expect("test database connection");
        diesel::update(platform_users::table.filter(platform_users::uuid.eq(viewer.id)))
            .set(platform_users::name.eq(Option::<String>::None))
            .execute(&mut conn)
            .expect("test target should support an absent optional name");
        drop(conn);

        assert!(matches!(
            create_project(
                &pool,
                viewer.id,
                org.id,
                "Unauthorized Project".to_string(),
                None,
            )
            .await,
            Err(PlatformResourceError::Unauthorized)
        ));
        assert!(matches!(
            update_membership(&pool, owner.id, org.id, owner.id, OrgRole::Admin, 4_096,).await,
            Err(PlatformResourceError::LastOwner)
        ));

        let promoted = update_membership(&pool, owner.id, org.id, viewer.id, OrgRole::Owner, 4_096)
            .await
            .expect("owner should promote a second owner");
        assert_eq!(promoted.role, "owner");
        assert!(promoted.name.is_none());
        let demoted = update_membership(&pool, owner.id, org.id, owner.id, OrgRole::Admin, 4_096)
            .await
            .expect("one of two owners may be demoted");
        assert_eq!(demoted.role, "admin");
        let memberships = list_memberships(&pool, viewer.id, org.id, 4_096)
            .await
            .expect("remaining owner should list memberships");
        assert_eq!(memberships.len(), 2);
        assert!(memberships
            .iter()
            .any(|membership| membership.user_id == viewer.id && membership.role == "owner"));
    }

    #[tokio::test]
    #[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
    async fn database_owner_invites_require_owner_issuer_and_verified_atomic_acceptance() {
        let Some(pool) = disposable_pool() else {
            return;
        };
        let mut fixture = PlatformFixture::new(pool.clone());
        let owner = fixture.actor("invite-owner");
        let admin = fixture.actor("invite-admin");
        let recipient = fixture.actor("invite-recipient");
        let org = create_org(&pool, owner.id, "Owner Invite Org".to_string())
            .await
            .expect("owner should create organization");
        fixture.track_org(org.id);
        fixture.add_member(org.id, admin.id, StrictRole::Admin);

        assert!(matches!(
            create_invite(
                &pool,
                admin.id,
                org.id,
                recipient.email.clone(),
                OrgRole::Owner,
                4_096,
            )
            .await,
            Err(PlatformResourceError::Unauthorized)
        ));
        let created = create_invite(
            &pool,
            owner.id,
            org.id,
            recipient.email.clone(),
            OrgRole::Owner,
            4_096,
        )
        .await
        .expect("owner should issue owner invite");
        let code = created.response.code;
        let active = list_invites(&pool, admin.id, org.id, 4_096)
            .await
            .expect("admin should list active invites");
        assert!(active.iter().any(|invite| invite.code == code));

        assert!(matches!(
            accept_invite(&pool, recipient.id, code, 4_096).await,
            Err(PlatformResourceError::VerifiedEmailRequired)
        ));
        let org_id = fixture.org_internal_id(org.id);
        let mut conn = pool.get().expect("test database connection");
        let membership_count = org_memberships::table
            .filter(org_memberships::org_id.eq(org_id))
            .filter(org_memberships::platform_user_id.eq(recipient.id))
            .count()
            .get_result::<i64>(&mut conn)
            .expect("membership count should load");
        let used = invite_codes::table
            .filter(invite_codes::code.eq(code))
            .select(invite_codes::used)
            .first::<bool>(&mut conn)
            .expect("invite should exist");
        assert_eq!(membership_count, 0);
        assert!(!used);

        diesel::insert_into(platform_email_verifications::table)
            .values((
                platform_email_verifications::platform_user_id.eq(recipient.id),
                platform_email_verifications::verification_code.eq(Uuid::new_v4()),
                platform_email_verifications::is_verified.eq(true),
                platform_email_verifications::expires_at.eq(Utc::now() + Duration::hours(1)),
            ))
            .execute(&mut conn)
            .expect("verified email marker should insert");
        drop(conn);

        accept_invite(&pool, recipient.id, code, 4_096)
            .await
            .expect("verified recipient should atomically accept owner invite");
        let mut conn = pool.get().expect("test database connection");
        let role = org_memberships::table
            .filter(org_memberships::org_id.eq(org_id))
            .filter(org_memberships::platform_user_id.eq(recipient.id))
            .select(org_memberships::role)
            .first::<String>(&mut conn)
            .expect("accepted membership should exist");
        let used = invite_codes::table
            .filter(invite_codes::code.eq(code))
            .select(invite_codes::used)
            .first::<bool>(&mut conn)
            .expect("accepted invite should exist");
        assert_eq!(role, "owner");
        assert!(used);
        drop(conn);
        assert!(matches!(
            accept_invite(&pool, recipient.id, code, 4_096).await,
            Err(PlatformResourceError::InviteAlreadyUsed)
        ));
    }
}
