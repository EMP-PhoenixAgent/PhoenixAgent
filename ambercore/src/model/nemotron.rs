//! NVIDIA Nemotron dense decoder — arch `nemotron` (Nemotron-3 / Nemotron-4
//! conversions that report the plain arch; the hybrid `nemotron_h` family is
//! deliberately unsupported, see `registry.rs`).
//!
//! Llama-shaped layout with two NVIDIA twists (matching llama.cpp's
//! `llama_model_nemotron` graph):
//!
//! - **LayerNorm (with bias)** instead of RMSNorm — `attn_norm.{weight,bias}`,
//!   `ffn_norm.{weight,bias}`, `output_norm.{weight,bias}`. The norm weights
//!   are dequantized to F32 (they are tiny).
//! - **Squared-ReLU FFN** — only `ffn_up` + `ffn_down` (no gate):
//!   `down(relu(up(x))²)`.
//!
//! RoPE is the llama-family convention (interleaved, GGUF-pre-permuted q/k).
//! MHA or GQA per `attention.head_count_kv`. The optional "Super" masked-
//! embedding / latent-FFN extras and conveyor vectors are vision-path only
//! and absent from text GGUFs; the loader fails cleanly if a variant needs
//! them.

use crate::error::{Error, Result};
use crate::model::common::{causal_mask, Meta, RopeTables};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::{DType, Device, Tensor};
use candle_nn::kv_cache::ConcatKvCache;
use candle_nn::{Embedding, LayerNorm, Module};
use candle_transformers::models::quantized_qwen3::Gguf;
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::utils::repeat_kv;

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
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> candle_core::Result<Tensor> {
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

/// LayerNorm loaded from a quantized pair of weight+bias tensors (F32 — small).
fn layernorm<R: std::io::Seek + std::io::Read>(
    gg: &mut Gguf<R>,
    name: &str,
    size: usize,
    eps: f64,
    device: &Device,
) -> candle_core::Result<LayerNorm> {
    let weight = gg.tensor(&format!("{name}.weight"))?.dequantize(device)?.to_dtype(DType::F32)?;
    let bias = gg.tensor(&format!("{name}.bias"))?.dequantize(device)?.to_dtype(DType::F32)?;
    if weight.dims() != [size] || bias.dims() != [size] {
        candle_core::bail!("{name} shape mismatch: {:?}/{:?}", weight.dims(), bias.dims());
    }
    Ok(LayerNorm::new(weight, bias, eps))
}

pub struct NemotronModel {
    arch: String,
    tok_embeddings: Embedding,
    layers: Vec<(LayerNorm, Attention, LayerNorm, QMatMul, QMatMul)>, // attn_norm, attn, ffn_norm, up, down
    norm: LayerNorm,
    output: QMatMul,
    device: Device,
    dtype: DType,
}

impl DynModel for NemotronModel {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .forward_inner(input, index_pos)
            .map_err(|e| Error::Model(format!("nemotron forward: {e}")))?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        for (_, attn, _, _, _) in self.layers.iter_mut() {
            attn.kv_cache.reset();
        }
    }
}

impl NemotronModel {
    fn forward_inner(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut xs = self.tok_embeddings.forward(input)?;
        let mask = if l > 1 { Some(causal_mask(&self.device, self.dtype, b, l, offset, None)?) } else { None };

        for (attn_norm, attn, ffn_norm, up, down) in self.layers.iter_mut() {
            let residual = &xs;
            let h = attn_norm.forward(&xs)?;
            let attn = attn.forward(&h, mask.as_ref(), offset)?;
            let xs_attn = (attn + residual)?;

            let residual = &xs_attn;
            let h = ffn_norm.forward(&xs_attn)?;
            // Squared ReLU: no gate tensor.
            let h = up.forward(&h)?;
            let h = h.apply(&candle_nn::Activation::Relu)?;
            let h = (&h * &h)?;
            let ffn = down.forward(&h)?;
            xs = (ffn + residual)?;
        }

        let xs = xs.narrow(1, l - 1, 1)?;
        let xs = self.norm.forward(&xs)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)
    }
}

/// Registry entry point: construct a Nemotron model from a loaded GGUF.
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
    let norm_eps = meta.opt_f32("attention.layer_norm_epsilon", 1e-5) as f64;
    let rope_freq_base = meta.opt_f32("rope.freq_base", 10_000.0) as f64;

    let inv_freq = crate::model::common::scaled_inv_freq(rope_freq_base, head_dim, crate::model::common::Scaling::None);
    let rope = std::sync::Arc::new(RopeTables::new(&inv_freq, context_length, DType::F32, device)?);

    let tok_embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?;
    let norm = layernorm(&mut gg, "output_norm", embedding_length, norm_eps, device)?;
    let output = match gg.qmatmul("output.weight") {
        Ok(v) => v,
        Err(_) => gg.qmatmul("token_embd.weight")?,
    };

    let mut layers = Vec::with_capacity(block_count);
    for i in 0..block_count {
        let prefix = format!("blk.{i}");
        let attn_norm = layernorm(&mut gg, &format!("{prefix}.attn_norm"), embedding_length, norm_eps, device)?;
        let ffn_norm = layernorm(&mut gg, &format!("{prefix}.ffn_norm"), embedding_length, norm_eps, device)?;
        let up = gg.qmatmul(&format!("{prefix}.ffn_up.weight"));
        let down = gg.qmatmul(&format!("{prefix}.ffn_down.weight"));
        // Guard the "Super" latent-FFN variant early: it replaces up/down with
        // ffn_latent_* + masked-embedding tensors AmberCore does not implement.
        let (up, down) = match (up, down) {
            (Ok(u), Ok(d)) => (u, d),
            _ => {
                return Err(Error::Model(format!(
                    "nemotron variant at blk.{i} lacks ffn_up/ffn_down (latent/masked-embedding \
                     'Super' variants are not supported)"
                )))
            }
        };
        layers.push((
            attn_norm,
            Attention {
                wq: gg.qmatmul(&format!("{prefix}.attn_q.weight"))?,
                wk: gg.qmatmul(&format!("{prefix}.attn_k.weight"))?,
                wv: gg.qmatmul(&format!("{prefix}.attn_v.weight"))?,
                wo: gg.qmatmul(&format!("{prefix}.attn_output.weight"))?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rope: rope.clone(),
                kv_cache: ConcatKvCache::new(2),
            },
            ffn_norm,
            up,
            down,
        ));
    }

    Ok(Box::new(NemotronModel {
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
    fn nemotron_end_to_end() {
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nemotron-test.gguf");

        let vocab = 64;
        let hidden = 16;
        let heads = 2;
        let head_dim = hidden / heads;
        let layers = 2;
        let ffn = 32;
        let meta = vec![
            ("general.architecture", Value::String("nemotron".into())),
            ("nemotron.embedding_length", Value::U32(hidden as u32)),
            ("nemotron.block_count", Value::U32(layers as u32)),
            ("nemotron.context_length", Value::U32(128)),
            ("nemotron.attention.head_count", Value::U32(heads as u32)),
            ("nemotron.attention.head_count_kv", Value::U32(1)),
            ("nemotron.attention.layer_norm_epsilon", Value::F32(1e-5)),
        ];
        let mut tensors: Vec<(String, Tensor)> = vec![
            ("token_embd.weight".to_string(), Tensor::randn(0f32, 1f32, (vocab, hidden), &dev).unwrap()),
            ("output_norm.weight".to_string(), Tensor::ones((hidden,), DType::F32, &dev).unwrap()),
            ("output_norm.bias".to_string(), Tensor::zeros((hidden,), DType::F32, &dev).unwrap()),
        ];
        for l in 0..layers {
            let p = format!("blk.{l}");
            for n in ["attn_norm", "ffn_norm"] {
                tensors.push((format!("{p}.{n}.weight"), Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
                tensors.push((format!("{p}.{n}.bias"), Tensor::zeros((hidden,), DType::F32, &dev).unwrap()));
            }
            tensors.push((format!("{p}.attn_q.weight"), Tensor::randn(0f32, 1f32, (hidden, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.attn_k.weight"), Tensor::randn(0f32, 1f32, (head_dim, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.attn_v.weight"), Tensor::randn(0f32, 1f32, (head_dim, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.attn_output.weight"), Tensor::randn(0f32, 1f32, (hidden, hidden), &dev).unwrap()));
            // NOTE: no ffn_gate — squared-ReLU FFN.
            tensors.push((format!("{p}.ffn_up.weight"), Tensor::randn(0f32, 1f32, (ffn, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.ffn_down.weight"), Tensor::randn(0f32, 1f32, (hidden, ffn), &dev).unwrap()));
        }
        crate::model::common::gguf_test_file(&path, &meta, &tensors).unwrap();

        let mut loaded = LoadedModel::load(&path).unwrap();
        assert_eq!(loaded.arch, "nemotron");
        let mut model = build(&mut loaded, &Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![2u32, 7, 1, 9], (1, 4), &Device::Cpu).unwrap();
        let logits = model.forward(&input, 0).unwrap();
        assert_eq!(logits.dims(), &[1, vocab]);
        model.clear_kv_cache();
    }
}
