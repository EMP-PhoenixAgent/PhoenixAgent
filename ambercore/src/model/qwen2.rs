//! Qwen2 architecture family (Qwen2 / Qwen2.5 and qwen2-layout relatives).
//!
//! Wraps [`candle_transformers::models::quantized_qwen2::ModelWeights`] — the
//! quantized (GGUF) Qwen2 implementation. Qwen2.5 reuses the Qwen2
//! architecture, so `"qwen2"` / `"qwen2_v2"` GGUFs route here directly.
//!
//! Two other families share Qwen2's exact tensor layout (`blk.N.attn_*`,
//! `ffn_gate/down/up`, NEOX rope) and differ only in their metadata namespace
//! — they load through the same candle model after a metadata-key remap:
//! - `"starcoder2"` (BigCode StarCoder2)
//! - `"internlm2"` (Shanghai AI Lab InternLM2)
//!
//! The model is built via `ModelWeights::from_gguf(content, &mut file, device)`,
//! which reads the quantized tensors off disk and constructs the layer stack.
//! Each [`forward`](DynModel::forward) call runs the transformer and returns
//! last-position logits `[batch, vocab]`.

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::mixtral::remap_metadata;
use crate::model::registry::DynModel;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2;

/// A constructed Qwen2-family model. Owns the quantized candle-transformers model.
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

/// Registry entry point: construct a Qwen2-family model from a loaded GGUF.
///
/// `starcoder2` / `internlm2` GGUFs get their metadata keys remapped into the
/// `qwen2.*` namespace first (their hyperparameters + tensor layout are
/// qwen2-compatible; only the namespace differs).
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let mut content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    match arch.as_str() {
        "starcoder2" | "internlm2" => {
            remap_metadata(&mut content, &arch, "qwen2");
            tracing::debug!("{arch}: remapped {arch}.* metadata keys to qwen2.*");
        }
        _ => {}
    }
    let inner = Qwen2::from_gguf(content, &mut loaded.file, device)
        .map_err(|e| Error::Model(format!("qwen2 from_gguf ({arch}): {e}")))?;
    Ok(Box::new(Qwen2Model { arch, inner }))
}
