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
///
/// Families:
/// - **Qwen**: `qwen2` (Qwen2/2.5), `qwen2_v2`, `qwen3`, `qwen3moe`
///   (Qwen3-30B-A3B & friends)
/// - **Llama family**: `llama` (Llama 1/2/3, Mistral-7B conversions,
///   TinyLlama, Yi, SmolLM, ...), `mixtral` (sparse MoE; via llama's MoE path)
/// - **Gemma**: `gemma`, `gemma2`, `gemma3`, `gemma4` (E2B/E4B/12B/31B dense,
///   PLE + cross-layer KV sharing + interleaved SWA)
/// - **Phi**: `phi2` (Phi-2), `phi3` (Phi-3 **and Phi-4**, which converts
///   with the phi3 arch)
/// - **GLM**: `glm4`
/// - **Liquid**: `lfm2`
/// - **Qwen2-layout relatives**: `starcoder2`, `internlm2` (metadata remap)
/// - **IBM Granite**: `granite` (dense), `granitemoe`, `granite_swa`
/// - **NVIDIA**: `nemotron` (dense LayerNorm/ReLU² family)
/// - **MiniMax**: `minimax-m2` (full-attention MoE)
/// - **Qwen 3.5**: `qwen35` — the hybrid gated-delta-net family (3-of-4
///   recurrent layers + every-4th full attention, chunked SSD prefill);
///   **covers Ornith**, which ships on the qwen35 archs
/// - **DeepSeek & relatives**: `deepseek2` (DeepSeek V2/V2.5/V3/R1 **plus
///   every model shipping that arch — Kimi K2 Instruct/Thinking, GLM-4.7
///   Lite, V2-Lite derivatives**) and `deepseek32` (V3.2; DSA indexer
///   ignored, full attention — llama.cpp's pre-DSA fallback mode). The legacy
///   `kimi_k2` spelling aliases `deepseek2` for older GGUFs.
///
/// Known-but-unsupported: **hybrid attention / SSM families** (recurrent
/// kernels candle lacks): `qwen35moe` (the MoE Qwen3.5 variant — no released
/// GGUF to validate against yet), `kimi-k3`, `kimi-linear`,
/// `nemotron_h`(+`_moe`), `granitehybrid`, `deepseek4`, `minimax-01`,
/// `qwen3next`, `graniteswitch`, `llama4`, `gpt-oss` (MXFP4), plus
/// encoder/diffusion archs (bert/t5/rwkv/dream...). Also unsupported:
/// `gemma3n` (AltUp/Laurel variant — queued behind gemma4) and gemma4-MoE
/// sizes (rejected at build with a clear message).
pub const SUPPORTED_ARCHS: &[&str] = &[
    "qwen2",
    "qwen2_v2",
    "qwen3",
    "qwen3moe",
    "qwen35",
    "rwkv7",
    "llama",
    "mixtral",
    "gemma",
    "gemma2",
    "gemma3",
    "gemma4",
    "phi2",
    "phi3",
    "glm4",
    "lfm2",
    "starcoder2",
    "internlm2",
    "granite",
    "granitemoe",
    "granite_swa",
    "nemotron",
    "minimax-m2",
    "deepseek2",
    "deepseek32",
    // Legacy alias: early Kimi-K2 GGUFs spelled the arch `kimi_k2` before
    // converters settled on deepseek2 (the layout is identical).
    "kimi_k2",
];

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
        "qwen2" | "qwen2_v2" | "starcoder2" | "internlm2" => {
            crate::model::qwen2::build(loaded, device)
        }
        "qwen3" => crate::model::qwen3::build(loaded, device),
        "qwen3moe" => crate::model::qwen3_moe::build(loaded, device),
        // Qwen3.5 hybrid GDN — NOT qwen3-compatible (ssm_* tensors, fused
        // attn_qkv, post_attention_norm, no ffn_norm); own builder.
        "qwen35" => crate::model::qwen35::build(loaded, device),
        // RWKV-7 "Goose" — pure-RNN linear attention (constant state per token;
        // no KV cache at all). Hand-built from llama.cpp's rwkv7 reference.
        "rwkv7" => crate::model::rwkv7::build(loaded, device),
        "llama" => crate::model::llama::build(loaded, device),
        "mixtral" => crate::model::mixtral::build(loaded, device),
        "gemma" | "gemma2" | "gemma3" => crate::model::gemma::build(loaded, device),
        "gemma4" => crate::model::gemma4::build(loaded, device),
        "phi2" | "phi3" => crate::model::phi::build(loaded, device),
        "glm4" => crate::model::glm4::build(loaded, device),
        "lfm2" => crate::model::lfm2::build(loaded, device),
        "granite" | "granitemoe" | "granite_swa" => crate::model::granite::build(loaded, device),
        "nemotron" => crate::model::nemotron::build(loaded, device),
        "minimax-m2" => crate::model::minimax_m2::build(loaded, device),
        // One MLA+MoE implementation covers DeepSeek V2/V2.5/V3/R1, V3.2
        // (`deepseek32`), Kimi K2 and the other deepseek2-arch models.
        "deepseek2" | "deepseek32" | "kimi_k2" => crate::model::deepseek2::build(loaded, device),
        other => Err(Error::Model(format!(
            "unsupported architecture: {other} (supported: {}; hybrid/SSM families like \
             kimi-k3, and nemotron_h are not supported yet — see registry.rs)",
            SUPPORTED_ARCHS.join(", ")
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
        // Spot-check each family.
        for arch in [
            "gemma3",
            "gemma4",
            "phi3",
            "glm4",
            "mixtral",
            "qwen3moe",
            "qwen35",
            "llama",
            "starcoder2",
            "granite",
            "granitemoe",
            "granite_swa",
            "nemotron",
            "minimax-m2",
            "deepseek2",
            "deepseek32",
            "kimi_k2",
        ] {
            assert!(is_supported(arch), "{arch} should be supported");
        }
        // Hybrid-SSM / exotic families — deliberately NOT supported (see the
        // SUPPORTED_ARCHS doc). Kimi K3 is `kimi-k3` (K2 rides deepseek2 and
        // IS supported); qwen35moe waits for a released GGUF to pin its names.
        for arch in [
            "qwen35moe",
            "deepseek4",
            "kimi-k3",
            "kimi-linear",
            "nemotron_h",
            "granitehybrid",
            "graniteswitch",
            "minimax-01",
            "minimax-m3",
            "gemma3n",
            "gpt-oss",
            "llama4",
        ] {
            assert!(!is_supported(arch), "{arch} must stay unsupported");
        }
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
