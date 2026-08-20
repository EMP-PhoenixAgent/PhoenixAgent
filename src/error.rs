//! Unified error types for Phoenix Agent.

use thiserror::Error;

/// Errors emitted by Phoenix Agent.
#[derive(Error, Debug)]
pub enum PhoenixError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("Migration error: {0}")]
    Migration(#[from] rusqlite_migration::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Crypto/KDF error: {0}")]
    Crypto(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Tool '{tool}' failed: {message}")]
    Tool { tool: String, message: String },

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, PhoenixError>;
