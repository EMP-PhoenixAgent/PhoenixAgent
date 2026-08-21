//! Phi architecture — Phi-2 (`phi2`) and Phi-3 / Phi-4 (`phi3`).
//!
//! Two candle implementations behind one module:
//! - `"phi2"` GGUFs → [`quantized_phi`] (Phi-2's fused-qkv layout).
//! - `"phi3"` GGUFs → [`quantized_phi3`]. Phi-4 and Phi-4-mini GGUFs also
//!   report `general.architecture = "phi3"` (llama.cpp converts them with the
//!   phi3 arch), so they load here too — including the long-rope scaling
//!   factors Phi-4 uses.
//!
//! Flash attention is not enabled (portable CPU/CUDA path; candle's fallback
//! matmul attention is used).
//!
//! KV cache: both loaders self-reset at `index_pos == 0` (phi3 uses an
//! explicit `candle_nn` KvCache it clears on prefill), so the
//! [`DynModel::clear_kv_cache`] default no-op is sufficient.

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_phi::ModelWeights as Phi2;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3;

/// A constructed Phi-2 model.
pub struct Phi2Model {
    arch: String,
    inner: Phi2,
}

impl DynModel for Phi2Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .inner
            .forward(input, index_pos)
            .map_err(|e| Error::Model(format!("phi2 forward: {e}")))?;
        Ok(logits)
    }
}

/// A constructed Phi-3/Phi-4 model.
pub struct Phi3Model {
    arch: String,
    inner: Phi3,
}

impl DynModel for Phi3Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .inner
            .forward(input, index_pos)
            .map_err(|e| Error::Model(format!("phi3 forward: {e}")))?;
        Ok(logits)
    }
}

/// Registry entry point: construct a Phi model from a loaded GGUF, picking the
/// Phi-2 or Phi-3/4 implementation from the architecture string.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    match arch.as_str() {
        "phi2" => {
            let inner = Phi2::from_gguf(content, &mut loaded.file, device)
                .map_err(|e| Error::Model(format!("phi2 from_gguf: {e}")))?;
            Ok(Box::new(Phi2Model { arch, inner }))
        }
        "phi3" => {
            let inner = Phi3::from_gguf(false, content, &mut loaded.file, device)
                .map_err(|e| Error::Model(format!("phi3 from_gguf: {e}")))?;
            Ok(Box::new(Phi3Model { arch, inner }))
        }
        other => Err(Error::Model(format!(
            "phi build: unexpected architecture {other} (expected phi2 or phi3)"
        ))),
    }
}
