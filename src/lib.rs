//! Phoenix Agent — a fully-local, encrypted, autonomous coding agent.
//!
//! Library crate: the binary ([`main.rs`]) is a thin shell over these modules.

pub mod agent;
pub mod backend;
pub mod config;
pub mod crypto;
pub mod db;
pub mod error;
pub mod health;
pub mod logging;
pub mod model;
pub mod web;
