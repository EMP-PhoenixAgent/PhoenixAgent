//! Error types for AmberCore.
//!
//! Single unified `Error` enum + `Result` alias used across the library, CLI,
//! and HTTP server. Variants grow as milestones land (M0 adds `Model`/`Backend`,
//! M2 adds `Server`, etc.).

use std::io;

/// The error type returned by all AmberCore operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("model error: {0}")]
    Model(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("server error: {0}")]
    Server(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, Error>;
