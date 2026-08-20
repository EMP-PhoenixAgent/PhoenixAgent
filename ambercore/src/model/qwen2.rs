//! Qwen2 / Qwen2.5 architecture (Phoenix's default model family).
//!
//! Wraps [`candle_transformers::models::quantized_qwen2::ModelWeights`] — the
//! quantized (GGUF) Qwen2 implementation. Qwen2.5 reuses the Qwen2 architecture,
//! so both `"qwen2"` and `"qwen2_v2"`-family GGUFs route here.
//!
//! The model is built via `ModelWeights::from_gguf(content, &mut file, device)`,
//! which reads the quantized tensors off disk and constructs the layer stack.
//! Each [`forward`](DynModel::forward) call runs the transformer and returns
//! logits; the pipeline samples the next token from them.

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;

/// A constructed Qwen2 model. Owns the quantized candle-transformers model.
pub struct Qwen2Model {
    arch: String,
    inner: Qwen2,
}

impl DynModel for Qwen2Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .inner
            .forward(input, index_pos)
            .map_err(|e| Error::Model(format!("qwen2 forward: {e}")))?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }
}

/// Registry entry point: construct a Qwen2 model from a loaded GGUF.
///
/// Consumes `loaded.content` and reads tensor data from `loaded.file`.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    let inner = Qwen2::from_gguf(content, &mut loaded.file, device)
        .map_err(|e| Error::Model(format!("qwen2 from_gguf: {e}")))?;
    Ok(Box::new(Qwen2Model { arch, inner }))
}
