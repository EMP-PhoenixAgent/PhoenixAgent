//! Google Gemma 4 — arch `gemma4` (E2B / E4B / 12B / 31B dense text models).
//!
//! The architecture that `gemma-4-E2B-it` GGUFs report. A gemma-3-style
//! sandwich-norm transformer with four additions, each mapped from llama.cpp's
//! `llama_model_gemma4` graph:
//!
//! - **Per-Layer Embeddings (PLE)** on the E* sizes: a second, per-layer
//!   embedding table (`per_layer_tok_embd`, `(vocab, pl_dim × n_layer)`) whose
//!   rows are looked up per token (kept **quantized** — the table is a large
//!   fraction of an E2B file — via `QTensor::embedding`), projected from the
//!   main hidden state (`per_layer_model_proj` + `per_layer_proj_norm`), and
//!   injected per layer: `x += post_norm(proj(gelu(inp_gate(x)) * ple_l))`.
//! - **Cross-layer KV sharing** (`attention.shared_kv_layers`): the trailing
//!   shared layers have no k/v projections at all and attend over an earlier
//!   layer's cache — the last full-attention KV layer's cache (shared full
//!   layers) or the one before it (shared SWA layers), exactly llama.cpp's
//!   `n_layer_kv_from_start - (is_swa ? 2 : 1)` mapping.
//! - **Dual head dims + dual rope**: interleaved SWA layers use
//!   `attention.key_length_swa` dims with `rope.freq_base_swa`; global layers
//!   use `attention.key_length` with `rope.freq_base` and an optional learned
//!   `blk.N.rope_freqs` frequency-factor table. RoPE is half-split (gemma
//!   convention); V gets a plain (weightless) RMSNorm.
//! - **Optional final logit softcapping** (`final_logit_softcapping`):
//!   `tanh(logits/c)·c`.
//!
//! Embeddings are scaled by `sqrt(hidden)` as in every Gemma. Q/K carry
//! per-head RMSNorm. The MoE variants (26B-A4B — `ffn_gate_inp` tensors) are
//! rejected with a clear message until their extra router-scale/norm path is
//! ported. `gemma4-assistant` (MTP draft models) is out of scope.

use crate::error::{Error, Result};
use crate::model::common::{causal_mask, KvSlot, Meta, RopeTables};
use crate::model::gguf::LoadedModel;
use crate::model::registry::DynModel;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::models::quantized_qwen3::Gguf;
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

/// A per-slot key/value cache. Owning layers append; shared layers read. The
/// `ConcatKvCache` type has no read-without-append, and the borrow checker
/// needs shared readers and one writer on the same table, so a tiny local
/// cache it is.
/// One attention block. Layers that own KV projections write their cache;
/// shared layers only read (the cache lives in the model's slot table).
struct Attention {
    wq: QMatMul,
    wk: Option<QMatMul>,
    wv: Option<QMatMul>,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    /// Index into the model-wide KV-slot table.
    slot: usize,
    rope: Arc<RopeTables>,
    swa: bool,
    /// `gemma4.attention.scale` override; `None` → `1/sqrt(head_dim)`.
    scale_override: Option<f64>,
}

struct Layer {
    attn_norm: RmsNorm,
    attn: Attention,
    attn_post_norm: RmsNorm,
    ffn_norm: RmsNorm,
    ffn_gate: QMatMul,
    ffn_up: QMatMul,
    ffn_down: QMatMul,
    ffn_post_norm: RmsNorm,
    /// PLE injection tensors (present only on PLE models).
    ple: Option<PleLayer>,
    /// Optional per-layer output scale (`blk.N.layer_output_scale`).
    out_scale: Option<f32>,
}

struct PleLayer {
    inp_gate: QMatMul, // (pl_dim, hidden)
    proj: QMatMul,     // (hidden, pl_dim)
    post_norm: RmsNorm,
}

/// Shared PLE state: the (quantized) per-layer embedding table and the
/// main-hidden projection prepared once per forward.
struct PleGlobal {
    per_layer_tok_embd: QTensor,
    model_proj: QMatMul, // (pl_dim * n_layer, hidden)
    proj_norm: RmsNorm,  // over pl_dim
    pl_dim: usize,
    n_layer: usize,
}

pub struct Gemma4Model {
    arch: String,
    tok_embeddings: Embedding,
    layers: Vec<Layer>,
    /// One cache per KV-owning layer (shared layers point into this table).
    kv_slots: Vec<KvSlot>,
    ple: Option<PleGlobal>,
    norm: RmsNorm,
    output: QMatMul,
    device: Device,
    dtype: DType,
    embedding_sqrt: f64,
    window: usize,
    softcap: Option<f64>,
    rms_eps: f64,
}

impl DynModel for Gemma4Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self
            .forward_inner(input, index_pos)
            .map_err(|e| Error::Model(format!("gemma4 forward: {e}")))?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        for slot in self.kv_slots.iter_mut() {
            slot.reset();
        }
    }
}

impl Gemma4Model {
    fn forward_inner(&mut self, input: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut xs = self.tok_embeddings.forward(input)?;
        xs = (xs * self.embedding_sqrt)?;

        // PLE: quantized row lookup → (b, l, n_layer, pl_dim), plus the
        // main-hidden projection mixed in (computed once, sliced per layer).
        let ple_ctx = match &self.ple {
            Some(ple) => {
                let embd = ple.per_layer_tok_embd.embedding(input)?; // (b*l, pl*n)
                let embd = (embd * (ple.pl_dim as f64).sqrt())?.reshape((b, l, ple.n_layer, ple.pl_dim))?;
                let proj = ple.model_proj.forward(&xs)?;
                let proj = proj.reshape((b, l, ple.n_layer, ple.pl_dim))?;
                let proj = (proj * (1.0 / (hidden_of(&xs) as f64).sqrt()))?;
                let proj = ple.proj_norm.forward(&proj)?;
                let mixed = (proj + embd)?;
                Some((mixed * (1.0 / (2.0f64).sqrt()))?)
            }
            None => None,
        };

        let causal = if l > 1 { Some(causal_mask(&self.device, self.dtype, b, l, offset, None)?) } else { None };

        for (il, layer) in self.layers.iter_mut().enumerate() {
            let residual = &xs;
            let h = layer.attn_norm.forward(&xs)?;

            let q = layer.attn.wq.forward(&h)?;
            let q = q.reshape((b, l, layer.attn.n_head, layer.attn.head_dim))?.transpose(1, 2)?.contiguous()?;
            let q_flat = q.flatten(0, 2)?;
            let q_flat = layer.attn.q_norm.forward(&q_flat)?;
            let q = q_flat.reshape((b, layer.attn.n_head, l, layer.attn.head_dim))?;
            let q = layer.attn.rope.apply_half(&q, self.dtype, offset)?;

            let (q, k, v) = match (&layer.attn.wk, &layer.attn.wv) {
                (Some(wk), Some(wv)) => {
                    let k = wk.forward(&h)?;
                    let v = wv.forward(&h)?;
                    let k = k.reshape((b, l, layer.attn.n_kv_head, layer.attn.head_dim))?.transpose(1, 2)?.contiguous()?;
                    let v = v.reshape((b, l, layer.attn.n_kv_head, layer.attn.head_dim))?.transpose(1, 2)?;
                    let k_flat = k.flatten(0, 2)?;
                    let k_flat = layer.attn.k_norm.forward(&k_flat)?;
                    let k = k_flat.reshape((b, layer.attn.n_kv_head, l, layer.attn.head_dim))?;
                    let k = layer.attn.rope.apply_half(&k, self.dtype, offset)?;
                    // Plain (weightless) RMSNorm on V — gemma4 quirk.
                    let v = rms_norm_weightless(v, self.rms_eps)?;

                    let cache = &mut self.kv_slots[layer.attn.slot];
                    let (k, v) = cache.append(&k, &v)?;
                    let k = repeat_kv(k, layer.attn.n_head / layer.attn.n_kv_head)?.contiguous()?;
                    let v = repeat_kv(v, layer.attn.n_head / layer.attn.n_kv_head)?.contiguous()?;
                    (q, k, v)
                }
                (None, None) => {
                    // Shared layer: reuse the paired slot's cache verbatim.
                    let (k, v) = self.kv_slots[layer.attn.slot].current()?;
                    let k = repeat_kv(k, layer.attn.n_head / layer.attn.n_kv_head)?.contiguous()?;
                    let v = repeat_kv(v, layer.attn.n_head / layer.attn.n_kv_head)?.contiguous()?;
                    (q, k, v)
                }
                _ => candle_core::bail!("gemma4: layer {il} has k or v projection but not both"),
            };

            let mask = if l > 1 {
                match layer.attn.swa && self.window > 0 {
                    true => Some(causal_mask(&self.device, self.dtype, b, l, offset, Some(self.window))?),
                    false => causal.clone(),
                }
            } else {
                None
            };

            let scale = layer.attn.scale_override.unwrap_or_else(|| 1.0 / (layer.attn.head_dim as f64).sqrt());
            let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if let Some(m) = &mask {
                let m = if m.dtype() != scores.dtype() { m.to_dtype(scores.dtype())? } else { m.clone() };
                scores = scores.broadcast_add(&m)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let attn = probs.matmul(&v)?;
            let attn = attn.transpose(1, 2)?.reshape((b, l, layer.attn.n_head * layer.attn.head_dim))?;
            let attn = layer.attn.wo.forward(&attn.to_dtype(self.dtype)?)?;
            let attn = layer.attn_post_norm.forward(&attn)?;
            let xs_attn = (attn + residual)?;

            let residual = &xs_attn;
            let h = layer.ffn_norm.forward(&xs_attn)?;
            let gate = layer.ffn_gate.forward(&h)?.apply(&candle_nn::Activation::GeluPytorchTanh)?;
            let up = layer.ffn_up.forward(&h)?;
            let ffn = layer.ffn_down.forward(&(gate * up)?)?;
            let ffn = layer.ffn_post_norm.forward(&ffn)?;
            let mut x = (ffn + residual)?;

            if let (Some(ple), Some(ctx)) = (&self.ple, &ple_ctx) {
                let ple_l = ctx.narrow(2, il, 1)?.reshape((b, l, ple.pl_dim))?;
                let g = layer.ple.as_ref().expect("PLE ctx without layer tensors");
                let gate = g.inp_gate.forward(&x)?.apply(&candle_nn::Activation::Gelu)?;
                let mixed = (gate * ple_l)?;
                let out = g.proj.forward(&mixed)?;
                let out = g.post_norm.forward(&out)?;
                x = (x + out)?;
            }

            if let Some(s) = layer.out_scale {
                x = (x * s as f64)?;
            }
            xs = x;
        }

        let xs = xs.narrow(1, l - 1, 1)?;
        let xs = self.norm.forward(&xs)?;
        let logits = self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)?;
        match self.softcap {
            // tanh via exp (candle has no tanh activation):
            // tanh(x) = 1 - 2 / (exp(2x) + 1), applied to logits/c then
            // rescaled by c.
            Some(c) => {
                let scaled = (&logits * (1.0 / c))?;
                let t = ((&scaled * 2.0)?.exp()? + 1.0)?;
                Ok(((1.0 - (2.0 / t)?)? * c)?)
            }
            None => Ok(logits),
        }
    }
}

fn hidden_of(x: &Tensor) -> usize {
    x.dims()[x.dims().len() - 1]
}

/// Weightless RMSNorm over the last dim (gemma4 normalizes V with it):
/// `x / sqrt(mean(x²) + eps)`, computed in F32.
fn rms_norm_weightless(x: Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let xs = x.to_dtype(DType::F32)?;
    let ms = xs.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let denom = (ms + eps)?.sqrt()?;
    let out = xs.broadcast_div(&denom)?;
    out.to_dtype(dtype)
}

/// Registry entry point: construct a Gemma-4 model from a loaded GGUF.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content()?;
    let arch = loaded.arch.clone();
    let mut gg = Gguf::new(content, &mut loaded.file, device.clone());
    let meta = Meta::new(gg.metadata(), &arch);

    let hidden = meta.req_u32("embedding_length")?;
    let block_count = meta.req_u32("block_count")?;
    let head_count = meta.req_u32("attention.head_count")?;
    let head_count_kv = meta.opt_u32("attention.head_count_kv", head_count);
    let context_length = meta.req_u32("context_length")?;
    let rms_eps = meta.opt_f32("attention.layer_norm_rms_epsilon", 1e-6) as f64;
    let head_dim_full = meta.opt_u32("attention.key_length", hidden / head_count);
    let head_dim_swa = meta.opt_u32("attention.key_length_swa", head_dim_full);
    let rope_base_full = meta.opt_f32("rope.freq_base", 1_000_000.0) as f64;
    let rope_base_swa = meta.opt_f32("rope.freq_base_swa", 10_000.0) as f64;
    let window = meta.opt_u32("attention.sliding_window", 0);
    let attn_scale = meta.opt_f32("attention.scale", 0.0) as f64;
    let softcap = meta.opt_f32("final_logit_softcapping", 0.0);
    let softcap = if softcap > 0.0 { Some(softcap as f64) } else { None };
    let shared_kv_layers = meta.opt_u32("attention.shared_kv_layers", 0);
    let kv_from_start = if shared_kv_layers > 0 { block_count - shared_kv_layers } else { block_count };
    let pl_dim = meta.opt_u32("embedding_length_per_layer_input", 0);
    let swa_pattern = meta.opt_bool_array("attention.sliding_window_pattern", block_count);

    let rope_full = Arc::new(RopeTables::new(
        &crate::model::common::scaled_inv_freq(rope_base_full, head_dim_full, crate::model::common::Scaling::None),
        context_length,
        DType::F32,
        device,
    )?);
    let rope_swa = Arc::new(RopeTables::new(
        &crate::model::common::scaled_inv_freq(rope_base_swa, head_dim_swa, crate::model::common::Scaling::None),
        context_length,
        DType::F32,
        device,
    )?);

    let tok_embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?;
    let norm = gg.rms_norm("output_norm.weight", rms_eps)?;
    let output = match gg.qmatmul("output.weight") {
        Ok(v) => v,
        Err(_) => gg.qmatmul("token_embd.weight")?,
    };

    let ple = if pl_dim > 0 {
        Some(PleGlobal {
            per_layer_tok_embd: gg.tensor("per_layer_token_embd.weight")?,
            model_proj: gg.qmatmul("per_layer_model_proj.weight")?,
            proj_norm: gg.rms_norm("per_layer_proj_norm.weight", rms_eps)?,
            pl_dim,
            n_layer: block_count,
        })
    } else {
        None
    };

    let mut layers = Vec::with_capacity(block_count);
    let mut n_slots = 0;
    for i in 0..block_count {
        let prefix = format!("blk.{i}");
        if gg.tensor(&format!("{prefix}.ffn_gate_inp.weight")).is_ok() {
            return Err(Error::Model(
                "gemma4 MoE variants (e.g. 26B-A4B) are not supported yet — dense E2B/E4B/12B/31B are".into(),
            ));
        }
        let is_swa = swa_pattern.as_ref().map(|p| p[i]).unwrap_or(false);
        let owns_kv = i < kv_from_start;
        let slot = if owns_kv {
            let s = n_slots;
            n_slots += 1;
            s
        } else if is_swa {
            kv_from_start.saturating_sub(2)
        } else {
            kv_from_start.saturating_sub(1)
        };
        let head_dim = if is_swa { head_dim_swa } else { head_dim_full };

        let out_scale = match gg.tensor(&format!("{prefix}.layer_output_scale")) {
            Ok(t) => {
                let v = t.dequantize(device)?.to_dtype(DType::F32)?.flatten_all()?.to_vec0::<f32>()?;
                Some(v)
            }
            Err(_) => None,
        };

        layers.push(Layer {
            attn_norm: gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_eps)?,
            attn: Attention {
                wq: gg.qmatmul(&format!("{prefix}.attn_q.weight"))?,
                wk: if owns_kv { Some(gg.qmatmul(&format!("{prefix}.attn_k.weight"))?) } else { None },
                wv: if owns_kv { Some(gg.qmatmul(&format!("{prefix}.attn_v.weight"))?) } else { None },
                wo: gg.qmatmul(&format!("{prefix}.attn_output.weight"))?,
                q_norm: gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_eps)?,
                k_norm: gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                slot,
                rope: if is_swa { rope_swa.clone() } else { rope_full.clone() },
                swa: is_swa,
                scale_override: if attn_scale != 0.0 { Some(attn_scale) } else { None },
            },
            attn_post_norm: gg.rms_norm(&format!("{prefix}.attn_post_norm.weight"), rms_eps)?,
            ffn_norm: gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_eps)?,
            ffn_gate: gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?,
            ffn_up: gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?,
            ffn_down: gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?,
            ffn_post_norm: gg.rms_norm(&format!("{prefix}.ffn_post_norm.weight"), rms_eps)?,
            ple: if ple.is_some() {
                Some(PleLayer {
                    inp_gate: gg.qmatmul(&format!("{prefix}.inp_gate.weight"))?,
                    proj: gg.qmatmul(&format!("{prefix}.proj.weight"))?,
                    post_norm: gg.rms_norm(&format!("{prefix}.post_norm.weight"), rms_eps)?,
                })
            } else {
                None
            },
            out_scale,
        });
    }

    Ok(Box::new(Gemma4Model {
        arch,
        tok_embeddings: Embedding::new(tok_embeddings, hidden),
        layers,
        kv_slots: (0..n_slots).map(|_| KvSlot::default()).collect(),
        ple,
        norm,
        output,
        device: device.clone(),
        dtype: DType::F32,
        embedding_sqrt: (hidden as f64).sqrt(),
        window,
        softcap,
        rms_eps,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file::Value;

    /// Write a tiny gemma4 GGUF. `ple` adds the per-layer embedding tensors,
    /// `shared_kv` makes the last layer a KV-less shared layer.
    fn write_test_gguf(path: &std::path::Path, ple: bool, shared_kv: bool) {
        let dev = Device::Cpu;
        let vocab = 64;
        let hidden = 16;
        let heads = 2;
        let head_dim = 8;
        let layers = 3;
        let ffn = 32;
        let pl_dim = 6;
        let kv_from_start = if shared_kv { layers - 1 } else { layers };
        let swa_pattern: Vec<Value> = vec![Value::Bool(true), Value::Bool(false), Value::Bool(true)];

        let mut meta = vec![
            ("general.architecture", Value::String("gemma4".into())),
            ("gemma4.embedding_length", Value::U32(hidden as u32)),
            ("gemma4.block_count", Value::U32(layers as u32)),
            ("gemma4.context_length", Value::U32(128)),
            ("gemma4.attention.head_count", Value::U32(heads as u32)),
            ("gemma4.attention.head_count_kv", Value::U32(1)),
            ("gemma4.attention.key_length", Value::U32(head_dim as u32)),
            ("gemma4.attention.key_length_swa", Value::U32(head_dim as u32)),
            ("gemma4.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("gemma4.attention.sliding_window", Value::U32(4)),
            ("gemma4.attention.sliding_window_pattern", Value::Array(swa_pattern)),
        ];
        if ple {
            meta.push(("gemma4.embedding_length_per_layer_input", Value::U32(pl_dim as u32)));
        }
        if shared_kv {
            meta.push(("gemma4.attention.shared_kv_layers", Value::U32((layers - kv_from_start) as u32)));
        }

        let mut tensors: Vec<(String, Tensor)> = vec![
            ("token_embd.weight".to_string(), Tensor::randn(0f32, 1f32, (vocab, hidden), &dev).unwrap()),
            ("output_norm.weight".to_string(), Tensor::ones((hidden,), DType::F32, &dev).unwrap()),
        ];
        if ple {
            tensors.push((
                "per_layer_token_embd.weight".to_string(),
                Tensor::randn(0f32, 1f32, (vocab, pl_dim * layers), &dev).unwrap(),
            ));
            tensors.push((
                "per_layer_model_proj.weight".to_string(),
                Tensor::randn(0f32, 1f32, (pl_dim * layers, hidden), &dev).unwrap(),
            ));
            tensors.push(("per_layer_proj_norm.weight".to_string(), Tensor::ones((pl_dim,), DType::F32, &dev).unwrap()));
        }
        for l in 0..layers {
            let p = format!("blk.{l}");
            let owns_kv = l < kv_from_start;
            for n in ["attn_norm", "attn_post_norm", "ffn_norm", "ffn_post_norm"] {
                tensors.push((format!("{p}.{n}.weight"), Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            }
            tensors.push((format!("{p}.attn_q.weight"), Tensor::randn(0f32, 1f32, (heads * head_dim, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.attn_q_norm.weight"), Tensor::ones((head_dim,), DType::F32, &dev).unwrap()));
            tensors.push((format!("{p}.attn_k_norm.weight"), Tensor::ones((head_dim,), DType::F32, &dev).unwrap()));
            if owns_kv {
                tensors.push((format!("{p}.attn_k.weight"), Tensor::randn(0f32, 1f32, (head_dim, hidden), &dev).unwrap()));
                tensors.push((format!("{p}.attn_v.weight"), Tensor::randn(0f32, 1f32, (head_dim, hidden), &dev).unwrap()));
            }
            tensors.push((format!("{p}.attn_output.weight"), Tensor::randn(0f32, 1f32, (hidden, heads * head_dim), &dev).unwrap()));
            tensors.push((format!("{p}.ffn_gate.weight"), Tensor::randn(0f32, 1f32, (ffn, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.ffn_up.weight"), Tensor::randn(0f32, 1f32, (ffn, hidden), &dev).unwrap()));
            tensors.push((format!("{p}.ffn_down.weight"), Tensor::randn(0f32, 1f32, (hidden, ffn), &dev).unwrap()));
            if ple {
                tensors.push((format!("{p}.inp_gate.weight"), Tensor::randn(0f32, 1f32, (pl_dim, hidden), &dev).unwrap()));
                tensors.push((format!("{p}.proj.weight"), Tensor::randn(0f32, 1f32, (hidden, pl_dim), &dev).unwrap()));
                tensors.push((format!("{p}.post_norm.weight"), Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            }
        }
        crate::model::common::gguf_test_file(path, &meta, &tensors).unwrap();
    }

    fn run(ple: bool, shared_kv: bool) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gemma4-test.gguf");
        write_test_gguf(&path, ple, shared_kv);
        let mut loaded = LoadedModel::load(&path).unwrap();
        assert_eq!(loaded.arch, "gemma4");
        let mut model = build(&mut loaded, &Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![1u32, 5, 2, 8], (1, 4), &Device::Cpu).unwrap();
        let logits = model.forward(&input, 0).unwrap();
        assert_eq!(logits.dims(), &[1, 64]);
        let next = Tensor::from_vec(vec![9u32], (1, 1), &Device::Cpu).unwrap();
        let logits2 = model.forward(&next, 4).unwrap();
        assert_eq!(logits2.dims(), &[1, 64]);
        model.clear_kv_cache();
    }

    #[test]
    fn gemma4_dense_end_to_end() {
        run(false, false);
    }

    #[test]
    fn gemma4_with_per_layer_embeddings() {
        run(true, false);
    }

    #[test]
    fn gemma4_with_shared_kv_layers() {
        run(false, true);
    }

    #[test]
    fn gemma4_full_feature_stack() {
        run(true, true);
    }
}
