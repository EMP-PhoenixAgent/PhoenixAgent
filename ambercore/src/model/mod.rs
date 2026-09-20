//! Model loading and architecture dispatch.
//!
//! AmberCore is **architecture-agnostic**: a GGUF file's metadata names its
//! architecture (e.g. `qwen2`, `llama`, `gemma3`), and [`registry`] maps that
//! name to a constructor that builds the appropriate candle-transformers model
//! from the loaded tensors. New architectures are a registry entry, not a
//! rewrite.
//!
//! Flow: `gguf::LoadedModel::load(path)` → [`LoadedModel`] →
//! `registry::build(&mut loaded, &device)` → a boxed [`DynModel`] the pipeline
//! drives.
//!
//! Adapter patterns:
//! 1. **Direct** — thin [`DynModel`] wrappers over candle's
//!    `quantized_*::ModelWeights::from_gguf` (qwen2, qwen3, llama, gemma, phi,
//!    glm4, lfm2).
//! 2. **Remap** — families that share a candle-supported tensor layout but
//!    use their own metadata namespace get their keys remapped first
//!    (mixtral → llama's MoE path, starcoder2/internlm2 → qwen2).
//! 3. **Port** — when candle's model needs a local fix, a copied +
//!    modified implementation lives here ([`qwen3_moe`], which adds the
//!    KV-cache clear candle 0.11 lacks).
//! 4. **Hand-built** — architectures candle has no quantized loader for at
//!    all, built here from llama.cpp's reference graph over the same
//!    quantized primitives (QMatMul / grouped MoE GEMM / ConcatKvCache):
//!    [`granite`], [`nemotron`], [`minimax_m2`], [`gemma4`], [`deepseek2`].
//!    [`common`] (masks, metadata, rope) + [`moe`] (routed experts) hold the
//!    pieces they share.

pub mod common;
pub mod deepseek2;
pub mod gguf;
pub mod gemma;
pub mod gemma4;
pub mod glm4;
pub mod granite;
pub mod lfm2;
pub mod llama;
pub mod minimax_m2;
pub mod mixtral;
pub mod moe;
pub mod nemotron;
pub mod phi;
pub mod qwen2;
pub mod qwen3;
pub mod qwen3_moe;
pub mod qwen35;
pub mod registry;
pub mod rwkv7;
pub mod st_shim;

pub use gguf::LoadedModel;
pub use registry::{build, DynModel};
