//! IBM Granite family — `granite` (dense 3.x), `granitemoe` (3.0/3.1 MoE) and
//! `granite_swa` (4.x dense/MoE with interleaved sliding-window attention).
//!
//! Granite shares llama's tensor layout (`attn_q/k/v/output`, `ffn_gate/up/down`,
//! `ffn_*_exps` for MoE) but wraps the standard graph in IBM's **scale
//! multipliers** (read from GGUF metadata, applied exactly where llama.cpp's
//! `llama_model_granite` graph puts them):
//!
//! - embeddings × `granite.embedding_scale` (skipped when 0/absent),
//! - attention scores use `granite.attention.scale` instead of
//!   `1/sqrt(head_dim)` when set,
//! - each attention/FFN output is × `granite.residual_scale` **before** the
//!   residual add (skipped when 0/absent),
//! - logits ÷ `granite.logit_scale`.
//!
//! RoPE is the llama-family convention: the GGUF conversion pre-permutes q/k,
//! so rope is **interleaved** (`rope_i`), optionally scaled (linear/YaRN).
//!
//! `granite_swa` adds a per-layer sliding window read from
//! `attention.sliding_window_pattern` (true = windowed layer) + 
//! `attention.sliding_window`; windowed layers keep a full `ConcatKvCache` and
//! mask outside keys (simple + correct; llama.cpp's rotating SWA cache is a
//! memory optimization we don't need at AmberCore's working context sizes).
//!
//! KV cache: `ConcatKvCache` per layer; [`DynModel::clear_kv_cache`] resets
//! them for replica reuse (the cache never self-resets on prefill).

use crate::error::{Error, Result};
use crate::model::common::{causal_mask, parse_scaling, Meta, RopeTables};
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

/// One layer's attention. Standard GQA; the Granite attention scale is applied
/// by the caller (it belongs to the softmax, not the projections).
struct Attention {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope: std::sync::Arc<RopeTables>,
    kv_cache: ConcatKvCache,
}

impl Attention {
    fn new<R: std::io::Seek + std::io::Read>(
        gg: &mut Gguf<R>,
        prefix: &str,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        rope: std::sync::Arc<RopeTables>,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            wq: gg.qmatmul(&format!("{prefix}.attn_q.weight"))?,
            wk: gg.qmatmul(&format!("{prefix}.attn_k.weight"))?,
            wv: gg.qmatmul(&format!("{prefix}.attn_v.weight"))?,
            wo: gg.qmatmul(&format!("{prefix}.attn_output.weight"))?,
            n_head,
            n_kv_head,
            head_dim,
            rope,
            kv_cache: ConcatKvCache::new(2),
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize, kq_scale: f64) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        let q = q.reshape((b, seq_len, self.n_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;

        let q = self.rope.apply_interleaved(&q, dtype, offset)?;
        let k = self.rope.apply_interleaved(&k, dtype, offset)?;

        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?.contiguous()?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?.contiguous()?;

        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * kq_scale)?;
        if let Some(m) = mask {
            let m = if m.dtype() != scores.dtype() { m.to_dtype(scores.dtype())? } else { m.to_owned() };
            scores = scores.broadcast_add(&m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, seq_len, self.n_head * self.head_dim))?;
        self.wo.forward(&ctx.to_dtype(dtype)?)
    }
}

/// Dense SwiGLU FFN.
struct Mlp {
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        self.down.forward(&(candle_nn::ops::silu(&gate)? * up)?)
    }
}

/// MoE FFN: routed experts (softmax, renorm) + the Granite-style shared expert.
struct GraniteMoe {
    routed: RoutedMoe,
    shexp: Mlp,
}

impl GraniteMoe {
    fn forward(&self, x: &Tensor, is_prefill: bool) -> candle_core::Result<Tensor> {
        let routed = self.routed.forward(x, is_prefill)?;
        let shared = self.shexp.forward(x)?;
        routed + shared
    }
}

enum Ffn {
    Dense(Mlp),
    Moe(GraniteMoe),
}

impl Ffn {
    fn forward(&self, x: &Tensor, is_prefill: bool) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(m) => m.forward(x),
            Self::Moe(m) => m.forward(x, is_prefill),
        }
    }
}

struct Layer {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    ffn: Ffn,
    /// `Some(window)` on `granite_swa` sliding layers.
    window: Option<usize>,
}

/// The built Granite-family model.
pub struct GraniteModel {
    arch: String,
    tok_embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    device: Device,
    dtype: DType,
    embedding_scale: f64,
    attention_scale: f64,
    residual_scale: f64,
    logit_scale: f64,
    head_dim: usize,
}

impl DynModel for GraniteModel {
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

impl GraniteModel {
    fn forward_inner(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut xs = self.tok_embeddings.forward(input)?;
        if self.embedding_scale != 0.0 {
            xs = (xs * self.embedding_scale)?;
        }

        let kq_scale = if self.attention_scale != 0.0 { self.attention_scale } else { 1.0 / (self.head_dim as f64).sqrt() };

        // Plain causal mask computed once; windowed variants per distinct
        // window (both only needed when the chunk is longer than one token).
        let causal = if l > 1 {
            Some(causal_mask(&self.device, self.dtype, b, l, offset, None)?)
        } else {
            None
        };

        for layer in self.layers.iter_mut() {
            let mask = if l > 1 {
                match layer.window {
                    Some(w) => Some(causal_mask(&self.device, self.dtype, b, l, offset, Some(w))?),
                    None => causal.clone(),
                }
            } else {
                None
            };
            let residual = &xs;
            let h = layer.attn_norm.forward(&xs)?;
            let attn = layer.attn.forward(&h, mask.as_ref(), offset, kq_scale)?;
            let attn = if self.residual_scale != 0.0 { (attn * self.residual_scale)? } else { attn };
            let xs_attn = (attn + residual)?;

            let residual = &xs_attn;
            let h = layer.ffn_norm.forward(&xs_attn)?;
            let ffn = layer.ffn.forward(&h, l > 1)?;
            let ffn = if self.residual_scale != 0.0 { (ffn * self.residual_scale)? } else { ffn };
            xs = (ffn + residual)?;
        }

        let xs = xs.narrow(1, l - 1, 1)?;
        let xs = self.norm.forward(&xs)?;
        let logits = self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)?;
        if self.logit_scale != 0.0 && self.logit_scale != 1.0 {
            Ok((logits * (1.0 / self.logit_scale))?)
        } else {
            Ok(logits)
        }
    }
}

/// Registry entry point: construct a Granite-family model from a loaded GGUF.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    let mut gg = Gguf::new(content, &mut loaded.file, device.clone());
    let meta = Meta::new(gg.metadata(), &arch);

    let embedding_length = meta.req_u32("embedding_length")?;
    let block_count = meta.req_u32("block_count")?;
    let head_count = meta.req_u32("attention.head_count")?;
    let head_count_kv = meta.opt_u32("attention.head_count_kv", head_count);
    let head_dim = meta.opt_u32("attention.key_length", embedding_length / head_count);
    let context_length = meta.req_u32("context_length")?;
    let rms_norm_eps = meta.opt_f32("attention.layer_norm_rms_epsilon", 1e-5) as f64;
    let rope_freq_base = meta.opt_f32("rope.freq_base", 10_000.0) as f64;

    let embedding_scale = meta.opt_f32("embedding_scale", 0.0) as f64;
    let attention_scale = meta.opt_f32("attention.scale", 0.0) as f64;
    let residual_scale = meta.opt_f32("residual_scale", 0.0) as f64;
    let logit_scale = meta.opt_f32("logit_scale", 0.0) as f64;

    let expert_count = meta.opt_u32("expert_count", 0);
    let expert_used_count = meta.opt_u32("expert_used_count", 0);
    let expert_ff = meta.opt_u32("expert_feed_forward_length", 0);

    let window = meta.opt_u32("attention.sliding_window", 0);
    let swa_pattern = meta.opt_bool_array("attention.sliding_window_pattern", block_count);
    let scaling = parse_scaling(&meta);

    let inv_freq = crate::model::common::scaled_inv_freq(rope_freq_base, head_dim, scaling);
    let rope = std::sync::Arc::new(RopeTables::new(&inv_freq, context_length, DType::F32, device)?);

    let tok_embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?;
    let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
    let output = match gg.qmatmul("output.weight") {
        Ok(v) => v,
        Err(_) => gg.qmatmul("token_embd.weight")?, // tied embeddings
    };

    let mut layers = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let prefix = format!("blk.{i}");
        let is_moe = expert_count > 0 && gg.tensor(&format!("{prefix}.ffn_gate_inp.weight")).is_ok();
        let ffn = if is_moe {
            let routed = RoutedMoe::load(
                &mut gg,
                &prefix,
                embedding_length,
                expert_count,
                expert_ff,
                expert_used_count,
                candle_nn::Activation::Silu,
                Gating::Softmax,
                true, // llama.cpp build_moe_ffn(norm=true) for granite
                1.0,
                device,
            )?;
            let shexp = Mlp {
                gate: gg.qmatmul(&format!("{prefix}.ffn_gate_shexp.weight"))?,
                up: gg.qmatmul(&format!("{prefix}.ffn_up_shexp.weight"))?,
                down: gg.qmatmul(&format!("{prefix}.ffn_down_shexp.weight"))?,
            };
            Ffn::Moe(GraniteMoe { routed, shexp })
        } else {
            Ffn::Dense(Mlp {
                gate: gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?,
                up: gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?,
                down: gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?,
            })
        };
        layers.push(Layer {
            attn_norm: gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?,
            attn: Attention::new(&mut gg, &prefix, head_count, head_count_kv, head_dim, rope.clone())?,
            ffn_norm: gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?,
            ffn,
            window: match (&swa_pattern, window) {
                (Some(pattern), w) if w > 0 => pattern.get(i).copied().unwrap_or(false).then_some(w),
                _ => None,
            },
        });
    }

    Ok(Box::new(GraniteModel {
        arch,
        tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
        layers,
        norm,
        output,
        device: device.clone(),
        dtype: DType::F32,
        embedding_scale,
        attention_scale,
        residual_scale,
        logit_scale,
        head_dim,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file::Value;

    fn write_test_gguf(path: &std::path::Path, moe: bool) -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let vocab = 64;
        let hidden = 16;
        let heads = 2;
        let head_dim = hidden / heads;
        let layers = 2;
        let ffn = 32;
        let ctx = 128;

        let mut meta = vec![
            ("general.architecture", Value::String("granite".into())),
            ("general.name", Value::String("test-granite".into())),
            ("granite.embedding_length", Value::U32(hidden as u32)),
            ("granite.block_count", Value::U32(layers as u32)),
            ("granite.context_length", Value::U32(ctx as u32)),
            ("granite.attention.head_count", Value::U32(heads as u32)),
            ("granite.attention.head_count_kv", Value::U32(1)),
            ("granite.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("granite.attention.scale", Value::F32(0.08)),
            ("granite.embedding_scale", Value::F32(4.0)),
            ("granite.residual_scale", Value::F32(0.22)),
            ("granite.logit_scale", Value::F32(8.0)),
        ];
        if moe {
            meta.push(("granite.expert_count", Value::U32(4)));
            meta.push(("granite.expert_used_count", Value::U32(2)));
            meta.push(("granite.expert_feed_forward_length", Value::U32(ffn as u32)));
            meta.push(("granite.expert_shared_feed_forward_length", Value::U32(8)));
        }
        // NOTE: projection tensors are GGUF-oriented (out, in).
        let mut tensors: Vec<(String, Tensor)> = vec![
            ("token_embd.weight".to_string(), Tensor::randn(0f32, 1f32, (vocab, hidden), &dev)?),
            ("output_norm.weight".to_string(), Tensor::ones((hidden,), DType::F32, &dev)?),
        ];
        for l in 0..layers {
            let p = format!("blk.{l}");
            tensors.push((format!("{p}.attn_norm.weight"), Tensor::ones((hidden,), DType::F32, &dev)?));
            tensors.push((format!("{p}.attn_q.weight"), Tensor::randn(0f32, 1f32, (hidden, hidden), &dev)?));
            tensors.push((format!("{p}.attn_k.weight"), Tensor::randn(0f32, 1f32, (head_dim, hidden), &dev)?));
            tensors.push((format!("{p}.attn_v.weight"), Tensor::randn(0f32, 1f32, (head_dim, hidden), &dev)?));
            tensors.push((format!("{p}.attn_output.weight"), Tensor::randn(0f32, 1f32, (hidden, hidden), &dev)?));
            tensors.push((format!("{p}.ffn_norm.weight"), Tensor::ones((hidden,), DType::F32, &dev)?));
            if moe {
                let n_exp = 4usize;
                tensors.push((format!("{p}.ffn_gate_inp.weight"), Tensor::randn(0f32, 1f32, (n_exp, hidden), &dev)?));
                tensors.push((format!("{p}.ffn_gate_exps.weight"), Tensor::randn(0f32, 1f32, (n_exp, ffn, hidden), &dev)?));
                tensors.push((format!("{p}.ffn_up_exps.weight"), Tensor::randn(0f32, 1f32, (n_exp, ffn, hidden), &dev)?));
                tensors.push((format!("{p}.ffn_down_exps.weight"), Tensor::randn(0f32, 1f32, (n_exp, hidden, ffn), &dev)?));
                tensors.push((format!("{p}.ffn_gate_shexp.weight"), Tensor::randn(0f32, 1f32, (8, hidden), &dev)?));
                tensors.push((format!("{p}.ffn_up_shexp.weight"), Tensor::randn(0f32, 1f32, (8, hidden), &dev)?));
                tensors.push((format!("{p}.ffn_down_shexp.weight"), Tensor::randn(0f32, 1f32, (hidden, 8), &dev)?));
            } else {
                tensors.push((format!("{p}.ffn_gate.weight"), Tensor::randn(0f32, 1f32, (ffn, hidden), &dev)?));
                tensors.push((format!("{p}.ffn_up.weight"), Tensor::randn(0f32, 1f32, (ffn, hidden), &dev)?));
                tensors.push((format!("{p}.ffn_down.weight"), Tensor::randn(0f32, 1f32, (hidden, ffn), &dev)?));
            }
        }

        // gguf_test_file quantizes each f32 tensor to Q8_0 and writes the file.
        crate::model::common::gguf_test_file(path, &meta, &tensors)?;
        Ok(())
    }

    fn run_forward(moe: bool) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("granite-test.gguf");
        write_test_gguf(&path, moe).unwrap();
        let mut loaded = LoadedModel::load(&path).unwrap();
        assert_eq!(loaded.arch, "granite");
        let mut model = build(&mut loaded, &Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![3u32, 1, 4, 1], (1, 4), &Device::Cpu).unwrap();
        let logits = model.forward(&input, 0).unwrap();
        assert_eq!(logits.dims(), &[1, 64], "last-position logits over vocab");
        // Second call at the next offset exercises the KV-cache append path.
        let next = Tensor::from_vec(vec![5u32], (1, 1), &Device::Cpu).unwrap();
        let logits2 = model.forward(&next, 4).unwrap();
        assert_eq!(logits2.dims(), &[1, 64]);
        model.clear_kv_cache();
    }

    #[test]
    fn dense_granite_end_to_end() {
        run_forward(false);
    }

    #[test]
    fn moe_granite_end_to_end() {
        run_forward(true);
    }
}
