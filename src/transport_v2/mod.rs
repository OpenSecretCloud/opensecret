//! Dormant implementation core for the additive encrypted transport protocol.
//!
//! This module is intentionally compiled but not wired into [`crate::AppState`]
//! or any router yet. The next stacked change activates it after its wire bytes,
//! parsing, replay, expiry, and state-machine behavior are independently fixed
//! by tests.

#![allow(dead_code)]

mod crypto;
mod envelope;
mod session;
mod session_cache;

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
