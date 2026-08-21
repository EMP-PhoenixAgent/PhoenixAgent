//! GLM-4 architecture (Zhipu AI).
//!
//! Wraps [`candle_transformers::models::quantized_glm4::ModelWeights`] for
//! `"glm4"` GGUFs (GLM-4-9B and friends). The loader handles GLM's
//! partial-RoPE (first 25% of head dims rotated) and the
//! `[gMASK]<sop>`-style prompt is produced by the chat-template layer
//! (`crate::tokenizer`), not here.
//!
//! KV cache: candle's glm4 loader self-resets when a prefill runs at
//! `offset == 0` (verified in candle 0.11), so the
//! [`DynModel::clear_kv_cache`] default no-op is sufficient.

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_glm4::ModelWeights as Glm4;

/// A constructed GLM-4 model.
pub struct Glm4Model {
    arch: String,
    inner: Glm4,
}

impl DynModel for Glm4Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .inner
            .forward(input, index_pos)
            .map_err(|e| Error::Model(format!("glm4 forward: {e}")))?;
        Ok(logits)
    }
}

/// Registry entry point: construct a GLM-4 model from a loaded GGUF.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    let inner = Glm4::from_gguf(content, &mut loaded.file, device, DType::F32)
        .map_err(|e| Error::Model(format!("glm4 from_gguf: {e}")))?;
    Ok(Box::new(Glm4Model { arch, inner }))
}
