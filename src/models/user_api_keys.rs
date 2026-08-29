use crate::models::schema::{user_api_keys, users};
use crate::models::users::User;
use chrono::{DateTime, Utc};
use diesel::dsl::{count_star, sql};
use diesel::prelude::*;
use diesel::sql_types::{BigInt, Nullable};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;
use zeroize::{Zeroize, Zeroizing};

#[derive(Error, Debug)]
pub enum UserApiKeyError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
    #[error("API key with this name already exists")]
    DuplicateName,
    #[error("API key not found")]
    NotFound,
    #[error("API key list output is too large")]
    OutputTooLarge,
    #[error("API key list changed during its bounded read")]
    InconsistentSnapshot,
}

#[derive(Queryable, Serialize, Deserialize, Clone)]
#[diesel(check_for_backend(diesel::pg::Pg))]
#[diesel(table_name = user_api_keys)]
pub struct UserApiKey {
    pub id: i32,
    pub user_id: Uuid,
    #[serde(skip_serializing)]
    pub key_hash: String,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// The authority resolved from a live API-key row and its current owning user.
///
/// This deliberately excludes the stored key hash so callers cannot retain or
/// accidentally log authentication material after the lookup completes.
#[derive(Clone)]
pub struct ResolvedUserApiKey {
    pub api_key_id: i32,
    pub user: User,
}

impl std::fmt::Debug for ResolvedUserApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedUserApiKey")
            .field("api_key_id", &self.api_key_id)
            .field("user_id", &self.user.uuid)
            .finish()
    }
}

/// Narrow metadata projection used only by bounded transport-v2 list reads.
#[derive(Queryable, Selectable)]
#[diesel(table_name = user_api_keys)]
pub struct UserApiKeyListItem {
    pub name: String,
    pub created_at: DateTime<Utc>,
}

impl Zeroize for UserApiKeyListItem {
    fn zeroize(&mut self) {
        self.name.zeroize();
    }
}

impl Drop for UserApiKeyListItem {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl std::fmt::Debug for UserApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserApiKey")
            .field("id", &self.id)
            .field("user_id", &self.user_id)
            .field("key_hash", &"<redacted>")
            .field("name", &self.name)
            .field("created_at", &self.created_at)
            .field("updated_at", &self.updated_at)
            .finish()
    }
}

#[derive(Insertable)]
#[diesel(table_name = user_api_keys)]
pub struct NewUserApiKey {
    pub user_id: Uuid,
    pub key_hash: String,
    pub name: String,
}

impl std::fmt::Debug for NewUserApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewUserApiKey")
            .field("user_id", &self.user_id)
            .field("key_hash", &"<redacted>")
            .field("name", &self.name)
            .finish()
    }
}

impl NewUserApiKey {
    pub fn new(user_id: Uuid, key_hash: String, name: String) -> Self {
        Self {
            user_id,
            key_hash,
            name,
        }
    }

    pub fn insert(self, conn: &mut PgConnection) -> Result<UserApiKey, UserApiKeyError> {
        diesel::insert_into(user_api_keys::table)
            .values(&self)
            .get_result(conn)
            .map_err(|e| match e {
                diesel::result::Error::DatabaseError(
                    diesel::result::DatabaseErrorKind::UniqueViolation,
                    ref info,
                ) if info.constraint_name() == Some("user_api_keys_user_id_name_key") => {
                    UserApiKeyError::DuplicateName
                }
                _ => UserApiKeyError::DatabaseError(e),
            })
    }
}

impl UserApiKey {
    pub fn get_by_id(conn: &mut PgConnection, id: i32) -> Result<Option<Self>, UserApiKeyError> {
        user_api_keys::table
            .filter(user_api_keys::id.eq(id))
            .first::<Self>(conn)
            .optional()
            .map_err(UserApiKeyError::DatabaseError)
    }

    pub fn get_by_key_hash(
        conn: &mut PgConnection,
        key_hash: &str,
    ) -> Result<Option<Self>, UserApiKeyError> {
        user_api_keys::table
            .filter(user_api_keys::key_hash.eq(key_hash))
            .first::<Self>(conn)
            .optional()
            .map_err(UserApiKeyError::DatabaseError)
    }

    /// Resolve a canonical API-key hash and its current owner in one query.
    pub fn resolve_by_key_hash(
        conn: &mut PgConnection,
        canonical_key_hash: &str,
    ) -> Result<Option<ResolvedUserApiKey>, UserApiKeyError> {
        user_api_keys::table
            .inner_join(users::table.on(users::uuid.eq(user_api_keys::user_id)))
            .filter(user_api_keys::key_hash.eq(canonical_key_hash))
            .select((user_api_keys::id, users::all_columns))
            .first::<(i32, User)>(conn)
            .optional()
            .map(|resolved| {
                resolved.map(|(api_key_id, user)| ResolvedUserApiKey { api_key_id, user })
            })
            .map_err(UserApiKeyError::DatabaseError)
    }

    /// Revalidate that a previously bound API-key ID still belongs to the
    /// exact user and return that current user row in one query.
    pub fn revalidate_owner(
        conn: &mut PgConnection,
        lookup_api_key_id: i32,
        lookup_user_id: Uuid,
    ) -> Result<Option<User>, UserApiKeyError> {
        users::table
            .inner_join(user_api_keys::table.on(users::uuid.eq(user_api_keys::user_id)))
            .filter(user_api_keys::id.eq(lookup_api_key_id))
            .filter(user_api_keys::user_id.eq(lookup_user_id))
            .select(users::all_columns)
            .first::<User>(conn)
            .optional()
            .map_err(UserApiKeyError::DatabaseError)
    }

    pub fn get_all_for_user(
        conn: &mut PgConnection,
        user_id: Uuid,
    ) -> Result<Vec<Self>, UserApiKeyError> {
        user_api_keys::table
            .filter(user_api_keys::user_id.eq(user_id))
            .load::<Self>(conn)
            .map_err(UserApiKeyError::DatabaseError)
    }

    /// Load only bounded API-key metadata for transport v2.
    ///
    /// The row count and total name bytes are measured before the narrow
    /// projection is loaded in the same read-only, repeatable-read snapshot.
    pub fn get_bounded_list_for_user(
        conn: &mut PgConnection,
        lookup_user_id: Uuid,
        logical_body_limit: usize,
        row_limit: usize,
    ) -> Result<Zeroizing<Vec<UserApiKeyListItem>>, UserApiKeyError> {
        conn.build_transaction()
            .read_only()
            .repeatable_read()
            .run::<_, UserApiKeyError, _>(|conn| {
                let (row_count, aggregate_name_bytes) = user_api_keys::table
                    .filter(user_api_keys::user_id.eq(lookup_user_id))
                    .select((
                        count_star(),
                        sql::<Nullable<BigInt>>("SUM(octet_length(name)::bigint)::bigint"),
                    ))
                    .first::<(i64, Option<i64>)>(conn)
                    .map_err(UserApiKeyError::DatabaseError)?;
                let (expected_rows, expected_name_bytes) = validate_bounded_list_aggregate(
                    row_count,
                    aggregate_name_bytes,
                    logical_body_limit,
                    row_limit,
                )?;

                let rows = Zeroizing::new(
                    user_api_keys::table
                        .filter(user_api_keys::user_id.eq(lookup_user_id))
                        .select(UserApiKeyListItem::as_select())
                        .load::<UserApiKeyListItem>(conn)
                        .map_err(UserApiKeyError::DatabaseError)?,
                );
                if rows.len() != expected_rows {
                    return Err(UserApiKeyError::InconsistentSnapshot);
                }
                let actual_name_bytes = rows.iter().try_fold(0_usize, |total, row| {
                    total
                        .checked_add(row.name.len())
                        .ok_or(UserApiKeyError::OutputTooLarge)
                })?;
                if actual_name_bytes != expected_name_bytes {
                    return Err(UserApiKeyError::InconsistentSnapshot);
                }
                Ok(rows)
            })
    }

    pub fn delete(self, conn: &mut PgConnection) -> Result<(), UserApiKeyError> {
        diesel::delete(user_api_keys::table.filter(user_api_keys::id.eq(self.id)))
            .execute(conn)
            .map(|_| ())
            .map_err(UserApiKeyError::DatabaseError)
    }

    pub fn delete_by_id(conn: &mut PgConnection, id: i32) -> Result<(), UserApiKeyError> {
        diesel::delete(user_api_keys::table.filter(user_api_keys::id.eq(id)))
            .execute(conn)
            .map(|_| ())
            .map_err(UserApiKeyError::DatabaseError)
    }

    pub fn delete_by_name_and_user(
        conn: &mut PgConnection,
        name: &str,
        user_id: Uuid,
    ) -> Result<(), UserApiKeyError> {
        let rows_affected = diesel::delete(
            user_api_keys::table
                .filter(user_api_keys::name.eq(name))
                .filter(user_api_keys::user_id.eq(user_id)),
        )
        .execute(conn)
        .map_err(UserApiKeyError::DatabaseError)?;

        if rows_affected == 0 {
            Err(UserApiKeyError::NotFound)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::schema::org_projects;
    use crate::models::users::NewUser;
    use sha2::Digest;

    #[test]
    #[ignore = "requires AEAD_TAMPER_TEST_DATABASE_URL pointing at disposable migrated local Postgres"]
    fn joined_resolution_and_live_owner_revalidation_fail_closed() {
        let Some(database_url) = std::env::var("AEAD_TAMPER_TEST_DATABASE_URL").ok() else {
            eprintln!("skipping: AEAD_TAMPER_TEST_DATABASE_URL is not set");
            return;
        };
        let mut conn = PgConnection::establish(&database_url)
            .expect("connect to disposable migrated PostgreSQL");

        conn.test_transaction::<_, diesel::result::Error, _>(|conn| {
            let project_id = org_projects::table
                .select(org_projects::id)
                .first::<i32>(conn)
                .expect("disposable database must contain an organization project");
            let user = NewUser::new(None, None, project_id)
                .insert(conn)
                .expect("insert API-key owner");
            let key_hash = hex::encode(sha2::Sha256::digest(b"v2-api-key-storage-test"));
            let key = NewUserApiKey::new(user.uuid, key_hash.clone(), "v2-test".to_owned())
                .insert(conn)
                .expect("insert API key");
            let key_id = key.id;

            let resolved = UserApiKey::resolve_by_key_hash(conn, &key_hash)
                .expect("joined authentication lookup")
                .expect("live key must resolve");
            assert_eq!(resolved.api_key_id, key_id);
            assert_eq!(resolved.user.uuid, user.uuid);
            assert!(UserApiKey::resolve_by_key_hash(conn, "missing")
                .unwrap()
                .is_none());

            assert_eq!(
                UserApiKey::revalidate_owner(conn, key_id, user.uuid)
                    .unwrap()
                    .map(|owner| owner.uuid),
                Some(user.uuid)
            );
            assert!(UserApiKey::revalidate_owner(conn, key_id, Uuid::new_v4())
                .unwrap()
                .is_none());

            key.delete(conn).expect("delete API key");
            assert!(UserApiKey::revalidate_owner(conn, key_id, user.uuid)
                .unwrap()
                .is_none());
            Ok(())
        });
    }
}

fn validate_bounded_list_aggregate(
    row_count: i64,
    aggregate_name_bytes: Option<i64>,
    logical_body_limit: usize,
    row_limit: usize,
) -> Result<(usize, usize), UserApiKeyError> {
    let row_count =
        usize::try_from(row_count).map_err(|_| UserApiKeyError::InconsistentSnapshot)?;
    if row_count > row_limit {
        return Err(UserApiKeyError::OutputTooLarge);
    }

    let aggregate_name_bytes = match (row_count, aggregate_name_bytes) {
        (0, None) => 0,
        (0, Some(_)) | (_, None) => return Err(UserApiKeyError::InconsistentSnapshot),
        (_, Some(bytes)) => {
            usize::try_from(bytes).map_err(|_| UserApiKeyError::InconsistentSnapshot)?
        }
    };
    if aggregate_name_bytes > logical_body_limit {
        return Err(UserApiKeyError::OutputTooLarge);
    }
    Ok((row_count, aggregate_name_bytes))
}

#[cfg(test)]
mod bounded_list_tests {
    use super::*;

    #[test]
    fn aggregate_preflight_bounds_rows_and_name_bytes() {
        assert_eq!(
            validate_bounded_list_aggregate(0, None, 0, 0).unwrap(),
            (0, 0)
        );
        assert_eq!(
            validate_bounded_list_aggregate(2, Some(9), 9, 2).unwrap(),
            (2, 9)
        );
        assert!(matches!(
            validate_bounded_list_aggregate(2, Some(9), 8, 2),
            Err(UserApiKeyError::OutputTooLarge)
        ));
        assert!(matches!(
            validate_bounded_list_aggregate(2, Some(9), 9, 1),
            Err(UserApiKeyError::OutputTooLarge)
        ));
        for inconsistent in [
            validate_bounded_list_aggregate(-1, None, 9, 2),
            validate_bounded_list_aggregate(1, None, 9, 2),
            validate_bounded_list_aggregate(0, Some(0), 9, 2),
            validate_bounded_list_aggregate(0, Some(1), 9, 2),
            validate_bounded_list_aggregate(1, Some(-1), 9, 2),
        ] {
            assert!(matches!(
                inconsistent,
                Err(UserApiKeyError::InconsistentSnapshot)
            ));
        }
    }
}
