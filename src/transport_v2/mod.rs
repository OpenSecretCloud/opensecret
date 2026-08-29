//! Additive encrypted transport-v2 protocol and isolated HTTP gateway.
//!
//! The gateway owns independent attestation/session state and never re-enters
//! the transport-v1 router. Application authentication and operation projection
//! are added in later stack layers over the independently tested protocol core.

#![allow(dead_code)]

mod application;
mod crypto;
mod envelope;
mod gateway;
mod session;
mod session_cache;
pub(crate) mod stored_conversations;
pub(crate) mod stored_resources;

pub(crate) use gateway::{router, TransportV2State};

pub(crate) const MAX_PENDING_ATTESTATIONS: usize = 65_536;
pub(crate) const MAX_LIVE_SESSIONS: usize = 65_536;

pub(crate) const V2_MEMORY_BUDGET_BYTES: usize = 256 * 1024 * 1024;
pub(crate) const ACCOUNTED_BYTES_PER_PENDING_ATTESTATION: usize = 192;
pub(crate) const ACCOUNTED_BYTES_PER_LIVE_SESSION: usize = 512;
pub(crate) const ACCOUNTED_BYTES_PER_REPLAY_ID: usize = 64;

pub(crate) const fn maximum_accounted_memory_bytes() -> usize {
    MAX_PENDING_ATTESTATIONS * ACCOUNTED_BYTES_PER_PENDING_ATTESTATION
        + MAX_LIVE_SESSIONS * ACCOUNTED_BYTES_PER_LIVE_SESSION
        + session::DEFAULT_GLOBAL_REPLAY_IDS * ACCOUNTED_BYTES_PER_REPLAY_ID
}

#[cfg(test)]
mod tests;
