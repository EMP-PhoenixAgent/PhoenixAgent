//! Qwen3 architecture.
//!
//! Wraps [`candle_transformers::models::quantized_qwen3::ModelWeights`] — the
//! quantized (GGUF) Qwen3 implementation (per-head q/k RMSNorm, no
//! self-resetting KV cache).
//!
//! **Qwen3.5 (`qwen35`) does NOT route here** — it is a hybrid SSM whose
//! tensor layout (`ssm_*`, fused `attn_qkv`, no `ffn_norm`) needs kernels
//! candle doesn't have; the registry rejects it cleanly as an unsupported
//! architecture (see the NOTE in [`registry`](super::registry)).
//!
//! The model is built via `ModelWeights::from_gguf(content, &mut file, device)`,
//! same shape as qwen2. Each [`forward`](DynModel::forward) call returns
//! last-position logits `[batch, vocab]`.

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;

/// A constructed Qwen3 model. Owns the quantized candle-transformers model.
pub struct Qwen3Model {
    arch: String,
    inner: Qwen3,
}

impl DynModel for Qwen3Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .inner
            .forward(input, index_pos)
            .map_err(|e| Error::Model(format!("qwen3 forward: {e}")))?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        // MANDATORY for qwen3: unlike qwen2 it does NOT self-reset at index_pos=0,
        // so reusing one instance across sessions without this leaks the prior
        // sequence's K/V (garbage + unbounded memory). See DynModel::clear_kv_cache.
        self.inner.clear_kv_cache();
    }
}

/// Registry entry point: construct a Qwen3 model from a loaded GGUF.
///
/// Consumes `loaded.content` and reads tensor data from `loaded.file`.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    let inner = Qwen3::from_gguf(content, &mut loaded.file, device)
        .map_err(|e| Error::Model(format!("qwen3 from_gguf: {e}")))?;
    Ok(Box::new(Qwen3Model { arch, inner }))
}
