//! LFM2 architecture (Liquid AI — Liquid Foundation Models, gen 2).
//!
//! Wraps [`candle_transformers::models::quantized_lfm2::ModelWeights`] for
//! `"lfm2"` GGUFs. LFM2 blocks couple a short fixed conv prefix with
//! bidirectional gated attention; candle handles the block plumbing, this
//! adapter only plugs it into the registry.
//!
//! KV cache: candle's lfm2 loader self-resets at `index_pos == 0` (verified in
//! candle 0.11), so the [`DynModel::clear_kv_cache`] default no-op suffices.

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_lfm2::ModelWeights as Lfm2;

/// A constructed LFM2 model.
pub struct Lfm2Model {
    arch: String,
    inner: Lfm2,
}

impl DynModel for Lfm2Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .inner
            .forward(input, index_pos)
            .map_err(|e| Error::Model(format!("lfm2 forward: {e}")))?;
        Ok(logits)
    }
}

/// Registry entry point: construct an LFM2 model from a loaded GGUF.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    let inner = Lfm2::from_gguf(content, &mut loaded.file, device)
        .map_err(|e| Error::Model(format!("lfm2 from_gguf: {e}")))?;
    Ok(Box::new(Lfm2Model { arch, inner }))
}
