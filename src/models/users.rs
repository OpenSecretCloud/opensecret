use crate::models::schema::users;
use chrono::{DateTime, Utc};
use diesel::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum UserError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] diesel::result::Error),
}

#[derive(QueryableByName, Queryable, AsChangeset, Serialize, Deserialize, Clone, PartialEq)]
#[diesel(check_for_backend(diesel::pg::Pg))]
#[diesel(table_name = users)]
pub struct User {
    id: i32,
    pub uuid: Uuid,
    pub name: Option<String>,
    pub email: Option<String>,
    pub password_enc: Option<Vec<u8>>,
    seed_enc: Option<Vec<u8>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl User {
    pub async fn get_seed_encrypted(&self) -> Option<Vec<u8>> {
        self.seed_enc.clone()
    }

    pub fn get_by_id(conn: &mut PgConnection, lookup_id: i32) -> Result<Option<User>, UserError> {
        users::table
            .filter(users::id.eq(lookup_id))
            .first::<User>(conn)
            .optional()
            .map_err(UserError::DatabaseError)
    }

    pub fn get_by_uuid(
        conn: &mut PgConnection,
        lookup_uuid: Uuid,
    ) -> Result<Option<User>, UserError> {
        users::table
            .filter(users::uuid.eq(lookup_uuid))
            .first::<User>(conn)
            .optional()
            .map_err(UserError::DatabaseError)
    }

    pub fn get_by_email(
        conn: &mut PgConnection,
        lookup_email: String,
    ) -> Result<Option<User>, UserError> {
        users::table
            .filter(users::email.eq(lookup_email))
            .first::<User>(conn)
            .optional()
            .map_err(UserError::DatabaseError)
    }

    pub fn set_key(
        &self,
        conn: &mut PgConnection,
        new_seed_encrypted: Vec<u8>,
    ) -> Result<(), UserError> {
        diesel::update(users::table)
            .filter(users::id.eq(self.id))
            .set(users::seed_enc.eq(new_seed_encrypted))
            .execute(conn)
            .map(|_| ())
            .map_err(UserError::DatabaseError)
    }

    pub fn get_id(&self) -> Uuid {
        self.uuid
    }

    pub fn get_email(&self) -> Option<&str> {
        self.email.as_deref()
    }

    pub fn update_password(
        &self,
        conn: &mut PgConnection,
        new_password_enc: Option<Vec<u8>>,
    ) -> Result<(), UserError> {
        diesel::update(users::table)
            .filter(users::id.eq(self.id))
            .set(users::password_enc.eq(new_password_enc))
            .execute(conn)
            .map(|_| ())
            .map_err(UserError::DatabaseError)
    }

    pub fn is_guest(&self) -> bool {
        self.email.is_none()
    }

    pub fn update(&self, conn: &mut PgConnection) -> Result<(), UserError> {
        diesel::update(users::table)
            .filter(users::id.eq(self.id))
            .set((
                users::email.eq(&self.email),
                users::password_enc.eq(&self.password_enc),
                users::name.eq(&self.name),
                users::updated_at.eq(diesel::dsl::now),
            ))
            .execute(conn)
            .map(|_| ())
            .map_err(UserError::DatabaseError)
    }
}

// Here we've implemented `Debug` manually to avoid accidentally logging the
// password hash.
impl std::fmt::Debug for User {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("User")
            .field("id", &self.id)
            .field("uuid", &self.uuid)
            .field("name", &self.name)
            .field("email", &self.email)
            .field("password", &"[redacted]")
            .field("private_key", &"[redacted]")
            .finish()
    }
}

#[derive(Insertable)]
#[diesel(table_name = users)]
pub struct NewUser {
    pub name: Option<String>,
    pub email: Option<String>,
    pub password_enc: Option<Vec<u8>>,
    pub seed_enc: Option<Vec<u8>>,
}

impl NewUser {
    pub fn new(email: Option<String>, password_enc: Option<Vec<u8>>) -> Self {
        NewUser {
            name: None,
            email,
            password_enc,
            seed_enc: None,
        }
    }

    pub fn insert(&self, conn: &mut PgConnection) -> Result<User, UserError> {
        diesel::insert_into(users::table)
            .values(self)
            .get_result::<User>(conn)
            .map_err(UserError::DatabaseError)
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn with_name_option(mut self, name: Option<String>) -> Self {
        self.name = name;
        self
    }
}

// Here we've implemented `Debug` manually to avoid accidentally logging the
// password hash.
impl std::fmt::Debug for NewUser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("User")
            .field("name", &self.name)
            .field("email", &self.email)
            .field("password_enc", &"[redacted]")
            .field("seed_enc", &"[redacted]")
            .finish()
    }
}
