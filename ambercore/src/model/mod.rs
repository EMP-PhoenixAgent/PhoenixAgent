//! Model loading and architecture dispatch.
//!
//! AmberCore is **architecture-agnostic**: a GGUF file's metadata names its
//! architecture (e.g. `qwen2`, `llama`), and [`registry`] maps that name to a
//! constructor that builds the appropriate candle-transformers model from the
//! loaded tensors. New architectures are a registry entry, not a rewrite.
//!
//! Flow: `gguf::LoadedModel::load(path)` → [`LoadedModel`] →
//! `registry::build(&mut loaded, &device)` → a boxed [`DynModel`] the pipeline
//! drives. M0 wires qwen2; M1+ adds llama and others.

pub mod gguf;
pub mod llama;
pub mod qwen2;
pub mod qwen3;
pub mod registry;

pub use gguf::LoadedModel;
pub use registry::{build, DynModel};
