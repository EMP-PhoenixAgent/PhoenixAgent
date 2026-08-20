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

/// Architectures [`build`] can construct — the single source of truth shared
/// by load-time dispatch and pull-time validation (Phoenix rejects a download
/// whose architecture isn't in this list before registering it).
pub const SUPPORTED_ARCHS: &[&str] = &["qwen2", "qwen2_v2", "qwen3", "llama"];

/// Whether an architecture string (a GGUF's `general.architecture`) can be
/// built by this registry.
pub fn is_supported(arch: &str) -> bool {
    SUPPORTED_ARCHS.contains(&arch)
}

/// Build a runnable model from a loaded GGUF, dispatching on its architecture.
///
/// Consumes the parsed GGUF [`Content`](candle_core::quantized::gguf_file::Content)
/// from `loaded` and reads tensor data from its file handle.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    match loaded.arch.as_str() {
        "qwen2" | "qwen2_v2" => crate::model::qwen2::build(loaded, device),
        // NOTE: `qwen35` (Qwen3.5 hybrid SSM) is NOT qwen3-compatible — its
        // tensor layout (`ssm_*`, fused `attn_qkv`, `post_attention_norm`, no
        // `ffn_norm`) needs kernels candle doesn't have. It must fail as
        // "unsupported architecture", not crash inside the qwen3 builder.
        "qwen3" => crate::model::qwen3::build(loaded, device),
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

    #[test]
    fn supported_archs_match_build_arms() {
        for arch in SUPPORTED_ARCHS {
            assert!(is_supported(arch), "{arch} should be supported");
        }
        // Qwen3.5 hybrid — deliberately NOT supported (see build's NOTE).
        assert!(!is_supported("qwen35"));
        assert!(!is_supported(""));
    }

    /// `probe_arch` must read a real GGUF header. This hand-crafts the smallest
    /// valid one: magic, version 2, 0 tensors, a single string KV
    /// `general.architecture = "qwen3"`.
    #[test]
    fn probe_arch_reads_minimal_gguf_header() {
        let mut buf = b"GGUF".to_vec();
        buf.extend(2u32.to_le_bytes()); // version
        buf.extend(0u64.to_le_bytes()); // tensor count
        buf.extend(1u64.to_le_bytes()); // metadata kv count
        put_gguf_str(&mut buf, "general.architecture");
        buf.extend(8u32.to_le_bytes()); // value type 8 = string
        put_gguf_str(&mut buf, "qwen3");

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("probe.gguf");
        std::fs::write(&path, &buf).expect("write");
        assert_eq!(crate::model::gguf::probe_arch(&path).unwrap(), "qwen3");
    }

    fn put_gguf_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend((s.len() as u64).to_le_bytes());
        buf.extend(s.as_bytes());
    }
}
