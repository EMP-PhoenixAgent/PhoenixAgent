//! DeepSeek V2/V2.5/V3/R1 — arch `deepseek2` — **and every other model that
//! ships with that arch**: Kimi K2 (Kimi-K2-Instruct/Thinking GGUFs report
//! `deepseek2`), GLM-4.7-Lite, and the V2-Lite derivatives. `deepseek32`
//! (DeepSeek V3.2, DSA sparse attention) loads through the same graph with the
//! indexer ignored (see below).
//!
//! **MLA — Multi-head Latent Attention.** Implemented in llama.cpp's
//! "absorbed" form, which is also the memory-cheap form:
//!
//! 1. `q = wq_b(rms(wq_a(x)))` (or plain `wq` on q-lora-less "lite" layers),
//!    split per head into `q_nope` (`key_length_mla - rope_dim`) and `q_pe`.
//! 2. `kv_a = wkv_a_mqa(x)` splits into the compressed latent `kv_cmpr`
//!    (`kv_lora_rank`, RMSNormed) and the shared `k_pe` rope part.
//! 3. `q_nope` is **absorbed** through the per-head `wk_b`
//!    (`q_nope @ wk_bᵀ → (kv_lora_rank)`), so attention runs against the
//!    latent: `Q = [q_absorbed | q_pe]`, `K = [kv_cmpr | k_pe]` (one shared
//!    KV "head"), `V = kv_cmpr`.
//! 4. Scores use `mscale² / sqrt(key_length_mla)` (the YaRN attention-scale
//!    compensation, `mscale = 1 + 0.1·yarn_log_mul·ln(factor)`).
//! 5. The context is **decompressed** per head through `wv_b` before `wo`.
//!
//! The KV cache stores the latent (`kv_lora_rank + rope_dim` per token) —
//! that is the point of MLA. `wk_b`/`wv_b` are dequantized to F32 (attention
//! tensors are a small fraction of these models; the batched head matmuls
//! need dense slices quantized tensors can't provide).
//!
//! RoPE is llama-family **interleaved** (`rope_i`) on the shared 64-dim part —
//! the GGUF conversion pre-permutes the projections, same as llama. YaRN
//! scaling (factor 40 for V3) adjusts the frequencies.
//!
//! **MoE**: sigmoid (V2.5+/V3/K2, read from `expert_gating_func`) or softmax
//! (V2) routing with optional `ffn_exp_probs_b` bias, top-k weight renorm per
//! `expert_weights_norm`, `expert_weights_scale` (V3's routed scaling factor),
//! a shared expert (`ffn_*_shexp`, sized `ff_exp × expert_shared_count`), and
//! `leading_dense_block_count` initial dense SwiGLU layers. DeepSeek V3's
//! group-corrected top-k is **not** implemented — llama.cpp doesn't either;
//! routing picks plain top-k.
//!
//! The first-k-dense + MTP/`nextn` layers: only trunk layers `0..block_count`
//! are loaded (GGUFs append the MTP block as extra tensors; speculative
//! decoding is out of scope).
//!
//! `deepseek32` (V3.2) adds a DSA indexer; it is skipped with a logged
//! warning and full attention is used — the same degraded-but-working mode
//! llama.cpp ran before its DSA implementation.

use crate::error::{Error, Result};
use crate::model::common::{causal_mask, parse_scaling, Meta, RopeTables, Scaling};
use crate::model::gguf::LoadedModel;
use crate::model::moe::{Gating, RoutedMoe};
use crate::model::registry::DynModel;
use candle_core::{DType, Device, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::{Embedding, Module};
use candle_transformers::models::quantized_qwen3::Gguf;
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

enum QProj {
    /// `wq_a` → `q_a_norm` → `wq_b` (q-lora path, most models).
    Lora { wq_a: QMatMul, q_a_norm: RmsNorm, wq_b: QMatMul },
    /// Direct `wq` ("lite" layers: V2-Lite, Kimi K2).
    Direct { wq: QMatMul },
}

struct MlaAttention {
    q: QProj,
    wkv_a_mqa: QMatMul,
    kv_a_norm: RmsNorm,
    wo: QMatMul,
    /// Dequantized per-head absorber `(n_head, kv_lora, qk_nope)`.
    wk_b: Tensor,
    /// Dequantized per-head decompressor `(n_head, kv_lora, v_dim)`.
    wv_b: Tensor,
    n_head: usize,
    qk_nope: usize,
    qk_rope: usize,
    kv_lora: usize,
    v_dim: usize,
    rope: std::sync::Arc<RopeTables>,
    kq_scale: f64,
    kv_cache: ConcatKvCache,
}

impl MlaAttention {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> candle_core::Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let dtype = x.dtype();

        // Q: low-rank or direct, then split per head.
        let q = match &self.q {
            QProj::Lora { wq_a, q_a_norm, wq_b } => {
                let qa = wq_a.forward(x)?;
                let qa = q_a_norm.forward(&qa)?;
                wq_b.forward(&qa)?
            }
            QProj::Direct { wq } => wq.forward(x)?,
        };
        let key_mla = self.qk_nope + self.qk_rope;
        let q = q.reshape((b, l, self.n_head, key_mla))?.transpose(1, 2)?; // (b, h, l, mla)
        let q_nope = q.narrow(3, 0, self.qk_nope)?.contiguous()?;
        let q_pe = q.narrow(3, self.qk_nope, self.qk_rope)?.contiguous()?;

        // KV latent + shared rope part.
        let kv_a = self.wkv_a_mqa.forward(x)?; // (b, l, kv_lora + rope)
        let kv_cmpr = self.kv_a_norm.forward(&kv_a.narrow(2, 0, self.kv_lora)?.contiguous()?)?;
        let k_pe = kv_a.narrow(2, self.kv_lora, self.qk_rope)?.unsqueeze(1)?.contiguous()?; // (b, 1, l, rope)

        // Interleaved rope (GGUF-pre-permuted, llama-family convention).
        let q_pe = self.rope.apply_interleaved(&q_pe, dtype, offset)?;
        let k_pe = self.rope.apply_interleaved(&k_pe, dtype, offset)?;

        // Absorb q_nope through wk_b → (b, h, l, kv_lora).
        let q_abs = q_nope.matmul(&self.wk_b.transpose(1, 2)?.unsqueeze(0)?)?; // broadcast over b

        // Q = [absorbed | rope], single shared K/V head over the latent.
        let q = Tensor::cat(&[&q_abs, &q_pe], 3)?.contiguous()?; // (b, h, l, kv+rope)
        let k = Tensor::cat(&[&kv_cmpr.unsqueeze(1)?, &k_pe], 3)?; // (b, 1, l, kv+rope)
        let v = kv_cmpr.unsqueeze(1)?; // (b, 1, l, kv_lora)

        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.n_head)?.contiguous()?;
        let v = repeat_kv(v, self.n_head)?.contiguous()?;

        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * self.kq_scale)?;
        if let Some(m) = mask {
            let m = if m.dtype() != scores.dtype() { m.to_dtype(scores.dtype())? } else { m.to_owned() };
            scores = scores.broadcast_add(&m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (b, h, l, kv_lora)

        // Decompress per head through wv_b, merge, project out.
        let ctx = ctx.matmul(&self.wv_b.unsqueeze(0)?)?; // (b, h, l, v_dim)
        let ctx = ctx.transpose(1, 2)?.reshape((b, l, self.n_head * self.v_dim))?;
        self.wo.forward(&ctx.to_dtype(dtype)?)
    }
}

enum Ffn {
    Dense { gate: QMatMul, up: QMatMul, down: QMatMul },
    Moe { routed: RoutedMoe, shexp: (QMatMul, QMatMul, QMatMul) },
}

impl Ffn {
    fn forward(&self, x: &Tensor, is_prefill: bool) -> candle_core::Result<Tensor> {
        match self {
            Ffn::Dense { gate, up, down } => {
                let g = gate.forward(x)?;
                let u = up.forward(x)?;
                down.forward(&(candle_nn::ops::silu(&g)? * u)?)
            }
            Ffn::Moe { routed, shexp: (gate, up, down) } => {
                let r = routed.forward(x, is_prefill)?;
                let g = gate.forward(x)?;
                let u = up.forward(x)?;
                let s = down.forward(&(candle_nn::ops::silu(&g)? * u)?)?;
                r + s
            }
        }
    }
}

struct Layer {
    attn_norm: RmsNorm,
    attn: MlaAttention,
    ffn_norm: RmsNorm,
    ffn: Ffn,
}

pub struct DeepSeek2Model {
    arch: String,
    tok_embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    device: Device,
    dtype: DType,
}

impl DynModel for DeepSeek2Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .forward_inner(input, index_pos)
            .map_err(|e| Error::Model(format!("{} forward: {e}", self.arch)))?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attn.kv_cache.reset();
        }
    }
}

impl DeepSeek2Model {
    fn forward_inner(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut xs = self.tok_embeddings.forward(input)?;
        let mask = if l > 1 { Some(causal_mask(&self.device, self.dtype, b, l, offset, None)?) } else { None };

        for layer in self.layers.iter_mut() {
            let residual = &xs;
            let h = layer.attn_norm.forward(&xs)?;
            let attn = layer.attn.forward(&h, mask.as_ref(), offset)?;
            let xs_attn = (attn + residual)?;

            let residual = &xs_attn;
            let h = layer.ffn_norm.forward(&xs_attn)?;
            let ffn = layer.ffn.forward(&h, l > 1)?;
            xs = (ffn + residual)?;
        }

        let xs = xs.narrow(1, l - 1, 1)?;
        let xs = self.norm.forward(&xs)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)
    }
}

/// Registry entry point: construct a DeepSeek-family model from a loaded GGUF.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    let mut gg = Gguf::new(content, &mut loaded.file, device.clone());
    let meta = Meta::new(gg.metadata(), &arch);

    let hidden = meta.req_u32("embedding_length")?;
    let block_count = meta.req_u32("block_count")?;
    let head_count = meta.req_u32("attention.head_count")?;
    let context_length = meta.req_u32("context_length")?;
    let rms_eps = meta.opt_f32("attention.layer_norm_rms_epsilon", 1e-6) as f64;
    let rope_freq_base = meta.opt_f32("rope.freq_base", 10_000.0) as f64;

    let q_lora_rank = meta.opt_u32("attention.q_lora_rank", 0);
    let kv_lora_rank = meta.req_u32("attention.kv_lora_rank")?;
    let key_length_mla = meta.opt_u32("attention.key_length_mla", 576);
    let value_length_mla = meta.opt_u32("attention.value_length_mla", 512);
    let qk_rope = meta.opt_u32("rope.dimension_count", 64).min(key_length_mla);
    let qk_nope = key_length_mla - qk_rope;

    let dense_lead = meta.opt_u32("leading_dense_block_count", 0);
    let expert_count = meta.opt_u32("expert_count", 0);
    let expert_used = meta.opt_u32("expert_used_count", 0);
    let expert_ff = meta.opt_u32("expert_feed_forward_length", 0);
    // The shared expert's intermediate size is ff_exp × expert_shared_count,
    // implied by the shexp tensors themselves.
    let weights_norm = meta.opt_bool("expert_weights_norm", false);
    let weights_scale = meta.opt_f32("expert_weights_scale", 1.0) as f64;
    let gating = match meta.opt_str("expert_gating_func").as_deref() {
        Some("sigmoid") => Gating::Sigmoid,
        // Default softmax keeps DeepSeek-V2-era GGUFs (no gating key) correct.
        _ => Gating::Softmax,
    };

    // YaRN: adjust frequencies + the mscale score compensation.
    let scaling = parse_scaling(&meta);
    let (factor, mscale) = match scaling {
        Scaling::Yarn { factor, .. } | Scaling::Linear { factor } => {
            let yarn_log_mul = meta.opt_f32("rope.yarn_log_mul", 0.1) / 0.1; // cancel the convert-script factor
            let mscale = 1.0 + 0.1 * yarn_log_mul as f64 * (factor as f64).ln();
            (factor as f64, mscale)
        }
        Scaling::None => (1.0, 1.0),
    };
    let _ = factor;
    let kq_scale = mscale * mscale / (key_length_mla as f64).sqrt();

    let inv_freq = crate::model::common::scaled_inv_freq(rope_freq_base, qk_rope, scaling);
    let rope = std::sync::Arc::new(RopeTables::new(&inv_freq, context_length, DType::F32, device)?);

    // DSA indexer (deepseek32): detected, warned about, ignored.
    if meta.has("attention.indexer.key_length") || gg.tensor("blk.0.indexer_proj.weight").is_ok() {
        tracing::warn!(
            arch = %arch,
            "DSA indexer tensors present but not implemented — running full attention \
             (llama.cpp's pre-DSA fallback mode; quality is close, long-context speed is not)"
        );
    }

    let tok_embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?;
    let norm = gg.rms_norm("output_norm.weight", rms_eps)?;
    let output = match gg.qmatmul("output.weight") {
        Ok(v) => v,
        Err(_) => gg.qmatmul("token_embd.weight")?,
    };

    let mut layers = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let prefix = format!("blk.{i}");

        let q = if q_lora_rank > 0 {
            QProj::Lora {
                wq_a: gg.qmatmul(&format!("{prefix}.attn_q_a.weight"))?,
                q_a_norm: gg.rms_norm(&format!("{prefix}.attn_q_a_norm.weight"), rms_eps)?,
                wq_b: gg.qmatmul(&format!("{prefix}.attn_q_b.weight"))?,
            }
        } else {
            QProj::Direct { wq: gg.qmatmul(&format!("{prefix}.attn_q.weight"))? }
        };

        // wk_b / wv_b: (n_head, kv_lora, qk_nope|v_dim) dequantized for the
        // batched head matmuls (quantized tensors can't be sliced per head).
        let wk_b = gg.tensor(&format!("{prefix}.attn_k_b.weight"))?.dequantize(device)?.to_dtype(DType::F32)?;
        let wv_b = gg.tensor(&format!("{prefix}.attn_v_b.weight"))?.dequantize(device)?.to_dtype(DType::F32)?;
        let expect_wk = [head_count, kv_lora_rank, qk_nope];
        if wk_b.dims() != expect_wk {
            return Err(Error::Model(format!(
                "deepseek2: blk.{i}.attn_k_b is {:?}, expected {:?} — legacy fused \
                 wkv_b GGUFs are not supported, re-convert with a current llama.cpp",
                wk_b.dims(), expect_wk
            )));
        }

        let ffn = if i < dense_lead {
            Ffn::Dense {
                gate: gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?,
                up: gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?,
                down: gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?,
            }
        } else {
            if expert_count == 0 || expert_used == 0 || expert_ff == 0 {
                return Err(Error::Model(format!(
                    "deepseek2: blk.{i} is past the dense lead but MoE metadata \
                     (expert_count/used/feed_forward_length) is missing"
                )));
            }
            let routed = RoutedMoe::load(
                &mut gg,
                &prefix,
                hidden,
                expert_count,
                expert_ff,
                expert_used,
                candle_nn::Activation::Silu,
                gating,
                weights_norm,
                weights_scale,
                device,
            )?;
            // The shared-expert intermediate size (ff_exp × shared_count) is
            // implied by the tensors themselves; a mismatch fails loudly at
            // the first matmul with candle's shape error.
            Ffn::Moe {
                routed,
                shexp: (
                    gg.qmatmul(&format!("{prefix}.ffn_gate_shexp.weight"))?,
                    gg.qmatmul(&format!("{prefix}.ffn_up_shexp.weight"))?,
                    gg.qmatmul(&format!("{prefix}.ffn_down_shexp.weight"))?,
                ),
            }
        };

        layers.push(Layer {
            attn_norm: gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_eps)?,
            attn: MlaAttention {
                q,
                wkv_a_mqa: gg.qmatmul(&format!("{prefix}.attn_kv_a_mqa.weight"))?,
                kv_a_norm: gg.rms_norm(&format!("{prefix}.attn_kv_a_norm.weight"), rms_eps)?,
                wo: gg.qmatmul(&format!("{prefix}.attn_output.weight"))?,
                wk_b,
                wv_b,
                n_head: head_count,
                qk_nope,
                qk_rope,
                kv_lora: kv_lora_rank,
                v_dim: value_length_mla,
                rope: rope.clone(),
                kq_scale,
                kv_cache: ConcatKvCache::new(2),
            },
            ffn_norm: gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_eps)?,
            ffn,
        });
    }

    Ok(Box::new(DeepSeek2Model {
        arch,
        tok_embeddings: Embedding::new(tok_embeddings, hidden),
        layers,
        norm,
        output,
        device: device.clone(),
        dtype: DType::F32,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file::Value;

    /// A tiny deepseek2: 1 dense-lead layer + 1 MoE layer, sigmoid routing
    /// with bias, shared expert, q-lora path, yarn-free.
    #[test]
    fn deepseek2_end_to_end() {
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("deepseek2-test.gguf");

        let vocab = 64;
        let hidden = 16;
        let heads = 2;
        let kv_lora = 12;
        let qk_nope = 8;
        let qk_rope = 4;
        let key_mla = qk_nope + qk_rope;
        let v_dim = 8;
        let q_lora = 10;
        let layers = 2;
        let n_exp: usize = 4;
        let ff_exp: usize = 24;
        let shexp_mult: usize = 1;
        let dense_ff = 32;

        let meta = vec![
            ("general.architecture", Value::String("deepseek2".into())),
            ("deepseek2.embedding_length", Value::U32(hidden as u32)),
            ("deepseek2.block_count", Value::U32(layers as u32)),
            ("deepseek2.context_length", Value::U32(128)),
            ("deepseek2.attention.head_count", Value::U32(heads as u32)),
            ("deepseek2.attention.q_lora_rank", Value::U32(q_lora as u32)),
            ("deepseek2.attention.kv_lora_rank", Value::U32(kv_lora as u32)),
            ("deepseek2.attention.key_length_mla", Value::U32(key_mla as u32)),
            ("deepseek2.attention.value_length_mla", Value::U32(v_dim as u32)),
            ("deepseek2.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("deepseek2.rope.dimension_count", Value::U32(qk_rope as u32)),
            ("deepseek2.leading_dense_block_count", Value::U32(1)),
            ("deepseek2.expert_count", Value::U32(n_exp as u32)),
            ("deepseek2.expert_used_count", Value::U32(2)),
            ("deepseek2.expert_feed_forward_length", Value::U32(ff_exp as u32)),
            ("deepseek2.expert_shared_count", Value::U32(shexp_mult as u32)),
            ("deepseek2.expert_weights_norm", Value::Bool(true)),
            ("deepseek2.expert_gating_func", Value::String("sigmoid".into())),
            ("deepseek2.feed_forward_length", Value::U32(dense_ff as u32)),
        ];

        let mut tensors: Vec<(String, Tensor)> = vec![
            ("token_embd.weight".to_string(), Tensor::randn(0f32, 1f32, (vocab, hidden), &dev).unwrap()),
            ("output_norm.weight".to_string(), Tensor::ones((hidden,), DType::F32, &dev).unwrap()),
        ];
        for l in 0..layers {
            let p = format!("blk.{l}");
            tensors.push((format!("{p}.attn_norm.weight"), Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            tensors.push((format!("{p}.attn_q_a.weight"), Tensor::randn(0f32, 1f32, (q_lora, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.attn_q_a_norm.weight"), Tensor::ones((q_lora,), DType::F32, &dev).unwrap()));
            tensors.push((format!("{p}.attn_q_b.weight"), Tensor::randn(0f32, 1f32, (heads * key_mla, q_lora), &dev).unwrap()));
            tensors.push((format!("{p}.attn_kv_a_mqa.weight"), Tensor::randn(0f32, 1f32, (kv_lora + qk_rope, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.attn_kv_a_norm.weight"), Tensor::ones((kv_lora,), DType::F32, &dev).unwrap()));
            // NOTE: wk_b GGUF ne = [qk_nope, kv_lora, n_head] → Rust (n_head, kv_lora, qk_nope).
            tensors.push((format!("{p}.attn_k_b.weight"), Tensor::randn(0f32, 1f32, (heads, kv_lora, qk_nope), &dev).unwrap()));
            tensors.push((format!("{p}.attn_v_b.weight"), Tensor::randn(0f32, 1f32, (heads, kv_lora, v_dim), &dev).unwrap()));
            tensors.push((format!("{p}.attn_output.weight"), Tensor::randn(0f32, 1f32, (hidden, heads * v_dim), &dev).unwrap()));
            tensors.push((format!("{p}.ffn_norm.weight"), Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            if l < 1 {
                tensors.push((format!("{p}.ffn_gate.weight"), Tensor::randn(0f32, 1f32, (dense_ff, hidden), &dev).unwrap()));
                tensors.push((format!("{p}.ffn_up.weight"), Tensor::randn(0f32, 1f32, (dense_ff, hidden), &dev).unwrap()));
                tensors.push((format!("{p}.ffn_down.weight"), Tensor::randn(0f32, 1f32, (hidden, dense_ff), &dev).unwrap()));
            } else {
                tensors.push((format!("{p}.ffn_gate_inp.weight"), Tensor::randn(0f32, 1f32, (n_exp, hidden), &dev).unwrap()));
                tensors.push((format!("{p}.ffn_exp_probs_b.bias"), Tensor::randn(0f32, 1f32, (n_exp,), &dev).unwrap()));
                tensors.push((format!("{p}.ffn_gate_exps.weight"), Tensor::randn(0f32, 1f32, (n_exp, ff_exp, hidden), &dev).unwrap()));
                tensors.push((format!("{p}.ffn_up_exps.weight"), Tensor::randn(0f32, 1f32, (n_exp, ff_exp, hidden), &dev).unwrap()));
                tensors.push((format!("{p}.ffn_down_exps.weight"), Tensor::randn(0f32, 1f32, (n_exp, hidden, ff_exp), &dev).unwrap()));
                tensors.push((format!("{p}.ffn_gate_shexp.weight"), Tensor::randn(0f32, 1f32, (ff_exp * shexp_mult, hidden), &dev).unwrap()));
                tensors.push((format!("{p}.ffn_up_shexp.weight"), Tensor::randn(0f32, 1f32, (ff_exp * shexp_mult, hidden), &dev).unwrap()));
                tensors.push((format!("{p}.ffn_down_shexp.weight"), Tensor::randn(0f32, 1f32, (hidden, ff_exp * shexp_mult), &dev).unwrap()));
            }
        }
        crate::model::common::gguf_test_file(&path, &meta, &tensors).unwrap();

        let mut loaded = LoadedModel::load(&path).unwrap();
        assert_eq!(loaded.arch, "deepseek2");
        let mut model = build(&mut loaded, &Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &Device::Cpu).unwrap();
        let logits = model.forward(&input, 0).unwrap();
        assert_eq!(logits.dims(), &[1, vocab]);
        // Decode step exercises the latent KV cache append + repeat_kv path.
        let next = Tensor::from_vec(vec![7u32], (1, 1), &Device::Cpu).unwrap();
        let logits2 = model.forward(&next, 4).unwrap();
        assert_eq!(logits2.dims(), &[1, vocab]);
        model.clear_kv_cache();
    }
}
