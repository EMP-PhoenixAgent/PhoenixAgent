//! Qwen3 / Qwen3.5 architecture.
//!
//! Wraps [`candle_transformers::models::quantized_qwen3::ModelWeights`] — the
//! quantized (GGUF) Qwen3 implementation. Qwen3.5 reuses the Qwen3 architecture,
//! so GGUFs reporting `general.architecture = "qwen35"` route here.
//!
//! The model is built via `ModelWeights::from_gguf(content, &mut file, device)`,
//! same shape as qwen2. Each [`forward`](DynModel::forward) call returns
//! last-position logits `[batch, vocab]`.
//!
//! [`candle_transformers::models::quantized_qwen3`]: https://docs.rs/candle-transformers/latest/candle_transformers/models/quantized_qwen3/index.html

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
///
/// **Qwen3.5 key remap:** Qwen3.5 GGUFs report `general.architecture = "qwen35"`
/// and namespace their metadata keys as `qwen35.*`, but candle's
/// `quantized_qwen3::ModelWeights::from_gguf` reads the `qwen3.*` prefix
/// (hardcoded). We remap `qwen35.` → `qwen3.` in the metadata map before
/// handing the content to candle, so Qwen3.5 GGUFs load correctly.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let mut content = loaded.take_content()?;
    let arch = loaded.arch.clone();

    // Remap qwen35.* → qwen3.* so candle's qwen3 builder finds its keys.
    if arch == "qwen35" {
        let remapped: Vec<(String, candle_core::quantized::gguf_file::Value)> = content
            .metadata
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("qwen35.")
                    .map(|suffix| (format!("qwen3.{suffix}"), v.clone()))
            })
            .collect();
        for (k, v) in remapped {
            content.metadata.insert(k, v);
        }
        tracing::debug!("qwen3.5: remapped qwen35.* metadata keys to qwen3.*");
    }

    let inner = Qwen3::from_gguf(content, &mut loaded.file, device)
        .map_err(|e| Error::Model(format!("qwen3 from_gguf: {e}")))?;
    Ok(Box::new(Qwen3Model { arch, inner }))
}
