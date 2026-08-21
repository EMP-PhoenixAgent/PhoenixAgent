//! Gemma architecture family — Gemma 1, Gemma 2, and Gemma 3.
//!
//! Wraps [`candle_transformers::models::quantized_gemma3::ModelWeights`]. That
//! single candle implementation probes the metadata prefixes
//! `gemma3` / `gemma2` / `gemma` (plus `gemma-embedding`) itself, so one
//! adapter serves every dense Gemma generation: `"gemma"`, `"gemma2"`, and
//! `"gemma3"` GGUFs all route here.
//!
//! Gemma quirks handled inside candle: embeddings are scaled by
//! `sqrt(hidden_size)`, attention uses query/key pre-feedback norms (gemma2+),
//! and Gemma 3 mixes local sliding-window layers with global-attention layers.
//!
//! KV cache: candle's gemma loader self-resets every layer's cache when a
//! prefill runs at `index_pos == 0` (verified in candle 0.11), so the
//! [`DynModel::clear_kv_cache`] default no-op is sufficient for replica reuse.

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma3::ModelWeights as Gemma;

/// A constructed Gemma-family model. Owns the quantized candle-transformers model.
pub struct GemmaModel {
    arch: String,
    inner: Gemma,
}

impl DynModel for GemmaModel {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .inner
            .forward(input, index_pos)
            .map_err(|e| Error::Model(format!("gemma forward: {e}")))?;
        Ok(logits)
    }
}

/// Registry entry point: construct a Gemma-family model from a loaded GGUF.
///
/// Consumes `loaded.content` and reads tensor data from `loaded.file`.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    let inner = Gemma::from_gguf(content, &mut loaded.file, device)
        .map_err(|e| Error::Model(format!("gemma from_gguf: {e}")))?;
    Ok(Box::new(GemmaModel { arch, inner }))
}
