//! MiniMax M2 — arch `minimax-m2` (MiniMax-M2 family; the older
//! lightning-attention `minimax-01` and the sparse-indexer `minimax-m3` are
//! separate archs, see `registry.rs`).
//!
//! M2 dropped MiniMax's lightning attention for a plain full-attention MoE
//! transformer that is structurally a qwen3-moe relative (llama.cpp's graph is
//! the same skeleton). Differences from AmberCore's ported
//! [`qwen3_moe`][crate::model::qwen3_moe]:
//!
//! - **Sigmoid routing with a gate bias** — `ffn_exp_probs_b.bias` (required
//!   in M2 GGUFs) added to the router logits before `sigmoid`, then top-8 with
//!   weight renormalization (`build_moe_ffn(norm=true)` in llama.cpp).
//! - **No shared expert** — every layer is routed; there are no
//!   `ffn_*_shexp` tensors.
//!
//! Kept from qwen3-moe: per-head q/k RMSNorm **before** rope, half-split
//! (NeoX) rope, GQA, `ConcatKvCache` with explicit clear for replica reuse.

use crate::error::{Error, Result};
use crate::model::common::{causal_mask, Meta};
use crate::model::gguf::LoadedModel;
use crate::model::moe::{Gating, RoutedMoe};
use crate::model::registry::DynModel;
use candle_core::{DType, Device, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::{Embedding, Module};
use candle_transformers::models::quantized_qwen3::{Gguf, RotaryEmbedding};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

struct Attention {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    dtype: DType,
    kv_cache: ConcatKvCache,
}

impl Attention {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        // M2 norms q over the FULL projected dim (all heads at once — unlike
        // qwen3's per-head norm) and k over all kv heads, before the split.
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q = q.reshape((1, seq_len, self.n_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((1, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((1, seq_len, self.n_kv_head, self.head_dim))?.transpose(1, 2)?;

        let (q, k) = (q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?);
        let (q, k) = self.rotary.apply(&q, &k, offset)?;

        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?.contiguous()?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
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

pub struct MiniMaxM2Model {
    arch: String,
    tok_embeddings: Embedding,
    layers: Vec<(RmsNorm, Attention, RmsNorm, RoutedMoe)>,
    norm: RmsNorm,
    output: QMatMul,
    device: Device,
    dtype: DType,
}

impl DynModel for MiniMaxM2Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .forward_inner(input, index_pos)
            .map_err(|e| Error::Model(format!("minimax-m2 forward: {e}")))?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        for (_, attn, _, _) in self.layers.iter_mut() {
            attn.kv_cache.reset();
        }
    }
}

impl MiniMaxM2Model {
    fn forward_inner(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut xs = self.tok_embeddings.forward(input)?;
        let mask = if l > 1 { Some(causal_mask(&self.device, self.dtype, b, l, offset, None)?) } else { None };

        for (attn_norm, attn, ffn_norm, moe) in self.layers.iter_mut() {
            let residual = &xs;
            let h = attn_norm.forward(&xs)?;
            let attn = attn.forward(&h, mask.as_ref(), offset)?;
            let xs_attn = (attn + residual)?;

            let residual = &xs_attn;
            let h = ffn_norm.forward(&xs_attn)?;
            let ffn = moe.forward(&h, l > 1)?;
            xs = (ffn + residual)?;
        }

        let xs = xs.narrow(1, l - 1, 1)?;
        let xs = self.norm.forward(&xs)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)
    }
}

/// Registry entry point: construct a MiniMax-M2 model from a loaded GGUF.
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
    let rope_freq_base = meta.opt_f32("rope.freq_base", 10_000.0);
    let expert_count = meta.req_u32("expert_count")?;
    let expert_used_count = meta.req_u32("expert_used_count")?;
    let expert_ff = meta.req_u32("expert_feed_forward_length")?;
    let weights_scale = meta.opt_f32("expert_weights_scale", 1.0) as f64;

    let rotary = Arc::new(RotaryEmbedding::new(DType::F32, head_dim, context_length, rope_freq_base as f64, device)?);

    let tok_embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?;
    let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
    let output = match gg.qmatmul("output.weight") {
        Ok(v) => v,
        Err(_) => gg.qmatmul("token_embd.weight")?,
    };

    let mut layers = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let prefix = format!("blk.{i}");
        let moe = RoutedMoe::load(
            &mut gg,
            &prefix,
            embedding_length,
            expert_count,
            expert_ff,
            expert_used_count,
            candle_nn::Activation::Silu,
            Gating::Sigmoid,
            true,
            weights_scale,
            device,
        )?;
        layers.push((
            gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?,
            Attention {
                wq: gg.qmatmul(&format!("{prefix}.attn_q.weight"))?,
                wk: gg.qmatmul(&format!("{prefix}.attn_k.weight"))?,
                wv: gg.qmatmul(&format!("{prefix}.attn_v.weight"))?,
                wo: gg.qmatmul(&format!("{prefix}.attn_output.weight"))?,
                q_norm: gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?,
                k_norm: gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary: rotary.clone(),
                dtype: DType::F32,
                kv_cache: ConcatKvCache::new(2),
            },
            gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?,
            moe,
        ));
    }

    Ok(Box::new(MiniMaxM2Model {
        arch,
        tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
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

    #[test]
    fn minimax_m2_end_to_end() {
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("minimax-m2-test.gguf");

        let vocab = 64;
        let hidden = 16;
        let heads = 2;
        let head_dim = 8;
        let layers = 2;
        let n_exp: usize = 4;
        let ff: usize = 32;
        let meta = vec![
            ("general.architecture", Value::String("minimax-m2".into())),
            ("minimax-m2.embedding_length", Value::U32(hidden as u32)),
            ("minimax-m2.block_count", Value::U32(layers as u32)),
            ("minimax-m2.context_length", Value::U32(128)),
            ("minimax-m2.attention.head_count", Value::U32(heads as u32)),
            ("minimax-m2.attention.head_count_kv", Value::U32(1)),
            ("minimax-m2.attention.key_length", Value::U32(head_dim as u32)),
            ("minimax-m2.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("minimax-m2.expert_count", Value::U32(n_exp as u32)),
            ("minimax-m2.expert_used_count", Value::U32(2)),
            ("minimax-m2.expert_feed_forward_length", Value::U32(ff as u32)),
        ];
        let mut tensors: Vec<(String, Tensor)> = vec![
            ("token_embd.weight".to_string(), Tensor::randn(0f32, 1f32, (vocab, hidden), &dev).unwrap()),
            ("output_norm.weight".to_string(), Tensor::ones((hidden,), DType::F32, &dev).unwrap()),
        ];
        for l in 0..layers {
            let p = format!("blk.{l}");
            tensors.push((format!("{p}.attn_norm.weight"), Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            tensors.push((format!("{p}.attn_q.weight"), Tensor::randn(0f32, 1f32, (heads * head_dim, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.attn_k.weight"), Tensor::randn(0f32, 1f32, (head_dim, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.attn_v.weight"), Tensor::randn(0f32, 1f32, (head_dim, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.attn_output.weight"), Tensor::randn(0f32, 1f32, (hidden, heads * head_dim), &dev).unwrap()));
            tensors.push((format!("{p}.attn_q_norm.weight"), Tensor::ones((heads * head_dim,), DType::F32, &dev).unwrap()));
            tensors.push((format!("{p}.attn_k_norm.weight"), Tensor::ones((head_dim,), DType::F32, &dev).unwrap()));
            tensors.push((format!("{p}.ffn_norm.weight"), Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            tensors.push((format!("{p}.ffn_gate_inp.weight"), Tensor::randn(0f32, 1f32, (n_exp, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.ffn_exp_probs_b.bias"), Tensor::randn(0f32, 1f32, (n_exp,), &dev).unwrap()));
            tensors.push((format!("{p}.ffn_gate_exps.weight"), Tensor::randn(0f32, 1f32, (n_exp, ff, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.ffn_up_exps.weight"), Tensor::randn(0f32, 1f32, (n_exp, ff, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.ffn_down_exps.weight"), Tensor::randn(0f32, 1f32, (n_exp, hidden, ff), &dev).unwrap()));
        }
        crate::model::common::gguf_test_file(&path, &meta, &tensors).unwrap();

        let mut loaded = LoadedModel::load(&path).unwrap();
        assert_eq!(loaded.arch, "minimax-m2");
        let mut model = build(&mut loaded, &Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &Device::Cpu).unwrap();
        let logits = model.forward(&input, 0).unwrap();
        assert_eq!(logits.dims(), &[1, vocab]);
        model.clear_kv_cache();
    }
}
