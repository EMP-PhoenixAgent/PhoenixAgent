//! Llama architecture — the largest GGUF family.
//!
//! Wraps [`candle_transformers::models::quantized_llama::ModelWeights`]. Any
//! GGUF reporting `general.architecture = "llama"` routes here: Llama 1/2/3
//! (and 3.x), Mistral 7B (llama.cpp converts it with the `llama` arch),
//! TinyLlama, Vicuna, CodeLlama, Yi, SmolLM — all share the llama tensor
//! layout. The candle loader also picks the RoPE convention (NEOX vs NORM)
//! from the architecture string and handles tied output weights.
//!
//! Sparse-MoE llama-family GGUFs (expert_count > 1) also load here — candle's
//! quantized_llama implements the router + per-expert FFN path, which is how
//! [`mixtral`](super::mixtral) loads via a metadata-key remap.
//!
//! Each [`forward`](DynModel::forward) call returns last-position logits
//! `[batch, vocab]`.

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights as Llama;

/// A constructed Llama-family model. Owns the quantized candle-transformers model.
pub struct LlamaModel {
    arch: String,
    inner: Llama,
}

impl DynModel for LlamaModel {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .inner
            .forward(input, index_pos)
            .map_err(|e| Error::Model(format!("llama forward: {e}")))?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        // candle's llama also self-resets each layer's cache when a prefill
        // runs at index_pos == 0; the explicit clear makes replica reuse safe
        // regardless of call pattern.
        self.inner.clear_kv_cache();
    }
}

/// Registry entry point: construct a Llama-family model from a loaded GGUF.
///
/// Consumes `loaded.content` and reads tensor data from `loaded.file`.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    let inner = Llama::from_gguf(content, &mut loaded.file, device)
        .map_err(|e| Error::Model(format!("llama from_gguf: {e}")))?;
    Ok(Box::new(LlamaModel { arch, inner }))
}
