//! Mixtral architecture — the sparse mixture-of-experts Llama family.
//!
//! Mixtral GGUFs report `general.architecture = "mixtral"` and namespace their
//! metadata as `mixtral.*`, but their **tensor layout is exactly llama's**
//! (plus the MoE tensors candle's `quantized_llama` already knows how to load:
//! the `ffn_gate_inp` router and the per-expert
//! `ffn_gate.{e}` / `ffn_down.{e}` / `ffn_up.{e}` FFNs, activated when
//! `expert_count > 1`). The rope convention is NORM (interleaved), which is
//! also what candle picks for anything not in its NEOX list — mixtral
//! included.
//!
//! So loading is a two-step trick, no new kernels:
//! 1. Remap the metadata keys `mixtral.*` → `llama.*` (hyperparameters only;
//!    tensor names already match).
//! 2. Hand the content to candle's `quantized_llama::ModelWeights::from_gguf`,
//!    which routes into its MoE path via the remapped `llama.expert_count`.
//!
//! Covers Mixtral 8x7B and 8x22B (and any other `mixtral.*` sparse-MoE GGUF).

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::quantized::gguf_file::{Content, Value};
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights as Llama;

/// A constructed Mixtral model. Owns the quantized candle-transformers model
/// (driven through its MoE path).
pub struct MixtralModel {
    arch: String,
    inner: Llama,
}

impl DynModel for MixtralModel {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .inner
            .forward(input, index_pos)
            .map_err(|e| Error::Model(format!("mixtral forward: {e}")))?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        self.inner.clear_kv_cache();
    }
}

/// Remap `{from}.*` metadata keys to `{to}.*` in place (hyperparameters only —
/// tensor names are not part of the metadata table).
pub(crate) fn remap_metadata(content: &mut Content, from: &str, to: &str) {
    let remapped: Vec<(String, Value)> = content
        .metadata
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix(&format!("{from}."))
                .map(|suffix| (format!("{to}.{suffix}"), v.clone()))
        })
        .collect();
    for (k, v) in remapped {
        content.metadata.insert(k, v);
    }
}

/// Registry entry point: construct a Mixtral model from a loaded GGUF via the
/// llama loader's MoE path.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let mut content = loaded.take_content()?;
    let arch = loaded.arch.clone();

    // mixtral.* → llama.* so candle's llama builder finds its hyperparameters
    // (head counts, expert_count, rope base, ...). Tensor names already match.
    remap_metadata(&mut content, "mixtral", "llama");
    tracing::debug!("mixtral: remapped mixtral.* metadata keys to llama.*");

    let inner = Llama::from_gguf(content, &mut loaded.file, device)
        .map_err(|e| Error::Model(format!("mixtral from_gguf (llama MoE path): {e}")))?;
    Ok(Box::new(MixtralModel { arch, inner }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_moves_mixtral_keys_to_llama_namespace() {
        let mut content = Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV2,
            metadata: [
                ("general.architecture".to_string(), Value::String("mixtral".into())),
                ("mixtral.block_count".to_string(), Value::U32(32)),
                ("mixtral.expert_count".to_string(), Value::U32(8)),
                ("llama.block_count".to_string(), Value::U32(1)), // pre-existing, overwritten
                ("qwen3.block_count".to_string(), Value::U32(99)), // untouched
            ]
            .into_iter()
            .collect(),
            tensor_infos: std::collections::HashMap::new(),
            tensor_data_offset: 0,
        };
        remap_metadata(&mut content, "mixtral", "llama");
        assert_eq!(
            content.metadata.get("llama.block_count").and_then(|v| v.to_u32().ok()),
            Some(32),
            "mixtral.block_count overwrites the llama key"
        );
        assert_eq!(
            content.metadata.get("llama.expert_count").and_then(|v| v.to_u32().ok()),
            Some(8),
            "the MoE switch lands where candle's llama loader reads it"
        );
        assert_eq!(
            content.metadata.get("qwen3.block_count").and_then(|v| v.to_u32().ok()),
            Some(99),
            "unrelated namespaces are untouched"
        );
        // The original mixtral.* keys remain (harmless; candle ignores them).
        assert!(content.metadata.contains_key("mixtral.expert_count"));
    }
}
