//! pullsift: cross-repo PR triage.
//!
//! Pure detection logic lives in the modules below and is unit-tested without
//! any network or database. The service wiring (webhook, GitHub client,
//! store) stays thin around it.

pub mod actions;
pub mod challenge;
pub mod cluster;
pub mod codeslop;
pub mod codestruct;
pub mod config;
pub mod diffsig;
pub mod dossier;
pub mod engine;
pub mod federation;
pub mod fit;
pub mod github;
pub mod hashing;
pub mod learn;
pub mod pipeline;
pub mod policy;
pub mod store;
pub mod stylometry;
pub mod textsig;
pub mod tokenscore;
pub mod webhook;
