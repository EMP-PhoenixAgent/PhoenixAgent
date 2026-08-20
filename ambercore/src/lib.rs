//! AmberCore, illuminated by Candle — a fully-Rust LLM runner.
//!
//! AmberCore loads quantized GGUF models and serves them over an HTTP API that
//! is wire-compatible with Ollama, so it can replace Ollama as the model
//! backend for the Phoenix Agent. The public API is **lib-first**: the HTTP
//! server is a thin binary wrapper, and a future milestone can compile AmberCore
//! directly into Phoenix as an in-process provider.
//!
//! See [`ACRoad.md`](https://example.invalid/ACRoad.md) for the full roadmap,
//! the Phoenix wire contract, and the design rationale.
//!
//! # Status
//!
//! v0.0.0 — scaffold. M0 (load GGUF → decode one token on CPU) is the next
//! implementation step. Every module compiles and carries documented
//! responsibilities + M-milestone markers for where its real logic lands.

pub mod backend;
pub mod catalog;
pub mod error;
pub mod model;
pub mod pipeline;
pub mod server;
pub mod tokenizer;

pub use backend::{Backend, CpuBackend};
pub use catalog::{Catalog, CatalogEntry};
pub use error::{Error, Result};
pub use model::{build, DynModel, LoadedModel};
pub use pipeline::{Pipeline, Token as GenToken};
pub use tokenizer::{Encoded, TokenizerWrapper};

/// The default port AmberCore's HTTP server binds to.
///
/// Phoenix Agent's `config.toml::ollama_url` defaults to Ollama's port `11434`;
/// to point Phoenix at AmberCore, set `ollama_url = "http://localhost:42069"`.
pub const DEFAULT_PORT: u16 = 42069;

/// AmberCore's version string.
pub const VERSION: &str = "0.0.0";
