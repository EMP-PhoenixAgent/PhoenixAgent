//! Architecture registry — maps an architecture string to a model constructor.
//!
//! The dispatch seam that keeps AmberCore architecture-agnostic. Each supported
//! architecture lives in its own module ([`qwen2`], [`llama`], ...) and
//! registers a builder here. `build()` reads the architecture name from a
//! [`LoadedModel`](super::gguf::LoadedModel) and hands it to the matching
//! builder.
//!
//! New architectures = one new module + one registry entry. The pipeline and
//! server never branch on architecture kind.

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;
use candle_core::{Device, Tensor};

/// A constructed, runnable model. The concrete candle-transformers type is
/// boxed here so the pipeline can drive any architecture uniformly.
///
/// `forward` takes the input token tensor `[batch, seq]` and the current
/// sequence position (used by the KV cache), and returns the **last-position**
/// logits as `[batch, vocab]` — i.e. the architecture is responsible for
/// slicing out the final token's logits before returning. (The quantized qwen2
/// implementation already does this internally.) The pipeline then squeezes the
/// batch dim and samples.
pub trait DynModel: Send {
    /// Architecture name this model was built from (e.g. `"qwen2"`).
    fn arch(&self) -> &str;

    /// Run a forward pass. `index_pos` is the absolute sequence position of the
    /// first token in `input` (the KV-cache offset). Returns `[batch, vocab]`.
    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor>;

    /// Reset the model's internal KV cache. Call between independent generation
    /// sessions on a *reused* model so a new prompt doesn't attend to the previous
    /// sequence's cached keys/values.
    ///
    /// **Why this exists:** candle's quantized models differ here — `quantized_qwen2`
    /// implicitly drops its cache whenever a prefill runs at `index_pos == 0`, but
    /// `quantized_qwen3` **appends unconditionally**, so reusing one qwen3 instance
    /// across sessions leaks the prior sequence's K/V (garbage output + unbounded
    /// memory growth). The pipeline calls this at the start of every `generate()`
    /// so both architectures start clean. Default no-op for stubs (e.g. llama).
    fn clear_kv_cache(&mut self) {}
}

/// Build a runnable model from a loaded GGUF, dispatching on its architecture.
///
/// Consumes the parsed GGUF [`Content`](candle_core::quantized::gguf_file::Content)
/// from `loaded` and reads tensor data from its file handle.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    match loaded.arch.as_str() {
        "qwen2" | "qwen2_v2" => crate::model::qwen2::build(loaded, device),
        // Qwen3 + Qwen3.5 share the qwen3 architecture.
        "qwen3" | "qwen35" => crate::model::qwen3::build(loaded, device),
        "llama" => crate::model::llama::build(loaded, device),
        other => Err(Error::Model(format!(
            "unsupported architecture: {other} (register it in src/model/registry.rs)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stub model proving the `clear_kv_cache` default is callable (no-op) so
    /// architectures that don't override it (e.g. the llama stub) still compile
    /// and satisfy the trait.
    struct StubModel;
    impl DynModel for StubModel {
        fn arch(&self) -> &str {
            "stub"
        }
        fn forward(&mut self, _input: &Tensor, _index_pos: usize) -> Result<Tensor> {
            Err(Error::Model("stub forward".into()))
        }
        // clear_kv_cache: inherited default no-op.
    }

    #[test]
    fn clear_kv_cache_default_is_callable_noop() {
        let mut m = StubModel;
        m.clear_kv_cache();
    }
}
