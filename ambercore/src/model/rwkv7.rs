//! RWKV-7 "Goose" — a linear-attention RNN family, hand-built (pattern #4)
//! from llama.cpp's reference graph (`src/models/rwkv7.cpp`,
//! `rwkv7-base.cpp`, and the `ggml_rwkv_wkv7` CPU kernel in ggml).
//!
//! Why RWKV-7 matters here: it is a pure RNN — the per-token cost and the
//! state size are **constant in sequence length** (state = `head_count ×
//! head_size²` f32 per layer, no KV cache at all), which makes small RWKV7
//! models ideal as a packed-within / default model on modest hardware.
//!
//! ## Architecture
//!
//! Every layer is:
//! 1. `x += time_mix(LN(x))` — the gated delta-rule linear attention.
//! 2. `x += channel_mix(LN(x))` — a ReLU² FFN with token shift.
//!
//! Token shift: each block consumes the previous token's post-LN activation
//! (`x_prev = shift(att_norm[0..T-1])`, zero for the first token) and mixes
//! per channel: `lerp(x, x_prev, t) = x + (x_prev − x) ⊙ t`.
//!
//! ## time_mix (per token t)
//!
//! Six per-channel lerp streams come from ONE fused tensor
//! (`time_mix_lerp_fused`, `[n_embd] × 6`): `xr, xw, xk, xv, xa, xg`.
//!
//! ```text
//! r = Wr·xr                       (receptance)
//! w = exp(−exp(−0.5) · σ(w0 + W2·tanh(W1·xw)))    (per-channel decay)
//! k = Wk·xk                       v = Wv·xv (+ first-layer value residual)
//! a = σ(a0 + A2·A1·xa)            g = G2·σ(G1·xg)   (gating, optional)
//! kk = l2norm_perhead(k ⊙ k_k)
//! k += (a − 1) ⊙ (k ⊙ k_a)
//! ```
//!
//! The WKV7 recurrence per head, in matrix form (verified against the ggml
//! CPU kernel's index arithmetic):
//!
//! ```text
//! sa = −S · kk                       (a row vector per head)
//! S ← diag(w)·S + v·kᵀ + sa·(kk ⊙ a)ᵀ   (in-place delta rule)
//! out = S · r
//! ```
//!
//! Then: per-head LayerNorm (no affine, eps 6.4e-4) → `⊙ ln_w + ln_b` →
//! `+= v ⊙ Σ_d(k·r·r_k)` (per-head scalar) → `⊙ g` (if gating) → `Wo`.
//!
//! ## channel_mix (per token t)
//!
//! ```text
//! k = (relu(Wk·lerp(ffn_norm, x_prev, lerp_k)))²
//! out = Wv·k
//! ```
//!
//! ## Recurrence strategy
//!
//! llama.cpp also evaluates WKV7 token-sequentially (no chunked form), so
//! prefill loops over tokens with tensor ops per step — same asymptotics as
//! the reference, fully vectorized within a token. The projections (which
//! dominate at 7B scale) run batched over the whole sequence first.
//!
//! ## Notes
//!
//! - All state + activations are F32 (RWKV7 is trained in high precision;
//!   llama.cpp keeps f32 states too). Weights stay quantized (QMatMul).
//! - The value-residual LoRA (`v0/v1/v2`) only exists for layers ≥ 1; layer 0
//!   defines `v_first`.
//! - Gating (`g1/g2`) is optional per model (`attention.gate_lora_rank`).

use std::io::{Read, Seek};

use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::LayerNorm;
use candle_transformers::models::quantized_qwen3::Gguf;
use candle_transformers::models::with_tracing::QMatMul;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::model::gguf::LoadedModel;

/// Element-wise sigmoid (candle 0.11 keeps it in `candle_nn::ops`).
fn sigmoid(t: &Tensor) -> candle_core::Result<Tensor> {
    candle_nn::ops::sigmoid(t)
}

/// `exp(−0.5)` — the fixed decay scale in the w computation.
const DECAY_SCALE: f64 = 0.606531;
/// Per-head LayerNorm epsilon after the WKV7 recurrence (llama.cpp: 64e-5).
const GROUP_NORM_EPS: f64 = 64e-5;
/// l2-norm epsilon for `kk` (llama.cpp: 1e-12).
const L2_EPS: f64 = 1e-12;

/// Row-wise l2 normalization over the last dim: `x / sqrt(Σx² + eps)`.
fn l2_norm_last(x: &Tensor) -> candle_core::Result<Tensor> {
    let sq = x.sqr()?;
    let sum = sq.sum(D::Minus1)?;
    let denom = (sum + L2_EPS)?.sqrt()?.unsqueeze(D::Minus1)?;
    x.broadcast_div(&denom)
}

/// LayerNorm without affine over the last dim (biased variance).
fn norm_no_affine(x: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let mean = x.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean(D::Minus1)?.unsqueeze(D::Minus1)?;
    let denom = (var + eps)?.sqrt()?;
    centered.broadcast_div(&denom)
}

/// Load a 1-D f32 vector tensor (norm weights, lerp coefficients, …).
fn vec1<R: Read + Seek>(gg: &mut Gguf<R>, name: &str, device: &Device) -> Result<Tensor> {
    let t = gg
        .tensor(name)
        .map_err(|e| Error::Model(format!("rwkv7: missing {name}: {e}")))?
        .dequantize(device)?
        .to_dtype(DType::F32)?;
    Ok(t)
}

/// Load a LayerNorm (weight + bias) pair as f32 tensors.
fn load_ln<R: Read + Seek>(
    gg: &mut Gguf<R>,
    w: &str,
    b: &str,
    eps: f64,
    device: &Device,
) -> Result<LayerNorm> {
    let weight = vec1(gg, w, device)?;
    let bias = vec1(gg, b, device)?;
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Load a quantized projection, with a clear error naming the tensor.
fn qm<R: Read + Seek>(gg: &mut Gguf<R>, name: &str) -> Result<QMatMul> {
    gg.qmatmul(name)
        .map_err(|e| Error::Model(format!("rwkv7: missing {name}: {e}")))
}

// ───────────────────────────── time mix ─────────────────────────────

struct TimeMix {
    /// Hidden size (`n_embd`).
    hidden: usize,
    head_count: usize,
    head_size: usize,

    // Big projections (quantized).
    key: QMatMul,
    value: QMatMul,
    receptance: QMatMul,
    output: QMatMul,

    // Dynamic decay (w): exp(−0.6065·σ(w0 + W2·tanh(W1·xw))).
    w0: Tensor,
    w1: QMatMul,
    w2: QMatMul,

    // In-context-learning-rate (a): σ(a0 + A2·A1·xa).
    a0: Tensor,
    a1: QMatMul,
    a2: QMatMul,

    // First-layer value residual (layers ≥ 1): σ(v0 + V2·V1·xv).
    v0: Option<Tensor>,
    v1: Option<QMatMul>,
    v2: Option<QMatMul>,

    // Output gating (optional): g = G2·σ(G1·xg).
    g1: Option<QMatMul>,
    g2: Option<QMatMul>,

    /// The six fused lerp coefficients, one `[hidden]` tensor per stream
    /// (xr, xw, xk, xv, xa, xg).
    lerp_fused: Vec<Tensor>,

    k_k: Tensor,
    k_a: Tensor,
    r_k: Tensor,

    // Per-head group-norm affine.
    ln_w: Tensor,
    ln_b: Tensor,

    /// The recurrent state `[head_count, head_size, head_size]` f32.
    state: Tensor,
}

impl TimeMix {
    fn reset(&mut self) -> candle_core::Result<()> {
        let device = self.state.device().clone();
        let dims = self.state.dims().to_vec();
        self.state = Tensor::zeros(dims, DType::F32, &device)?;
        Ok(())
    }

    /// `x` = post-LN activations `[T, hidden]`; `x_prev` = the same shifted by
    /// one token (leading zero for the first token). `v_first` is layer 0's
    /// value output — set by layer 0, consumed by every later layer.
    fn forward(
        &mut self,
        x: &Tensor,
        x_prev: &Tensor,
        v_first: &mut Option<Tensor>,
    ) -> Result<Tensor> {
        let t_len = x.dims()[0];
        let sx = x_prev.sub(x)?;

        // Six lerped streams.
        let lerp = |t: &Tensor, sx: &Tensor, x: &Tensor| -> candle_core::Result<Tensor> {
            x.broadcast_add(&sx.broadcast_mul(&t.unsqueeze(0)?)?)
        };
        let xr = lerp(&self.lerp_fused[0], &sx, x)?;
        let xw = lerp(&self.lerp_fused[1], &sx, x)?;
        let xk = lerp(&self.lerp_fused[2], &sx, x)?;
        let xv = lerp(&self.lerp_fused[3], &sx, x)?;
        let xa = lerp(&self.lerp_fused[4], &sx, x)?;

        let r = self.receptance.forward(&xr)?;
        // w = exp(−0.606531 · σ(w0 + W2·tanh(W1·xw)))
        let w = {
            let h = self.w1.forward(&xw)?.tanh()?;
            let h = self.w2.forward(&h)?;
            let h = h.broadcast_add(&self.w0.unsqueeze(0)?)?;
            sigmoid(&h)?.affine(-DECAY_SCALE, 0.0)?.exp()?
        };
        let k = self.key.forward(&xk)?;
        let v = self.value.forward(&xv)?;
        // First-layer value residual: v += (v_first − v) ⊙ σ(v0 + V2·V1·xv).
        let v = match (&self.v1, &self.v2, &self.v0, v_first.as_ref()) {
            (Some(v1), Some(v2), Some(v0), Some(vf)) => {
                let inner = v2.forward(&v1.forward(&xv)?)?;
                let mix = sigmoid(&inner.broadcast_add(&v0.unsqueeze(0)?)?)?;
                v.broadcast_add(&vf.sub(&v)?.broadcast_mul(&mix)?)?
            }
            _ => v,
        };
        if v_first.is_none() {
            *v_first = Some(v.clone());
        }
        // a = σ(a0 + A2·A1·xa)
        let a = {
            let inner = self.a2.forward(&self.a1.forward(&xa)?)?;
            sigmoid(&inner.broadcast_add(&self.a0.unsqueeze(0)?)?)?
        };

        let (hc, hs) = (self.head_count, self.head_size);
        // kk = per-head l2norm(k ⊙ k_k)
        let kk = k
            .broadcast_mul(&self.k_k.unsqueeze(0)?)?
            .reshape((t_len, hc, hs))?;
        let kk = l2_norm_last(&kk)?;
        // k += (a − 1) ⊙ (k ⊙ k_a)
        let ka = k.broadcast_mul(&self.k_a.unsqueeze(0)?)?;
        let k = k.broadcast_add(&a.broadcast_mul(&ka)?.sub(&ka)?)?;

        // Reshape to [T, head_count, head_size].
        let r3 = r.reshape((t_len, hc, hs))?;
        let w3 = w.reshape((t_len, hc, hs))?;
        let k3 = k.reshape((t_len, hc, hs))?;
        let v3 = v.reshape((t_len, hc, hs))?;
        let a3 = a.reshape((t_len, hc, hs))?;
        // b = kk ⊙ a rides along the delta correction.
        let b3 = kk.broadcast_mul(&a3)?;

        // ── the WKV7 recurrence (token-sequential, like llama.cpp) ──
        let mut s = self.state.clone(); // [hc, hs, hs]
        let mut out_rows = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let r_t = r3.narrow(0, t, 1)?.squeeze(0)?.contiguous()?; // [hc, hs]
            let w_t = w3.narrow(0, t, 1)?.squeeze(0)?.contiguous()?;
            let k_t = k3.narrow(0, t, 1)?.squeeze(0)?.contiguous()?;
            let v_t = v3.narrow(0, t, 1)?.squeeze(0)?.contiguous()?;
            let kk_t = kk.narrow(0, t, 1)?.squeeze(0)?.contiguous()?;
            let b_t = b3.narrow(0, t, 1)?.squeeze(0)?.contiguous()?;

            // sa[h,i] = −Σ_j kk[h,j] · s[h,i,j]
            let sa = s.broadcast_mul(&kk_t.unsqueeze(1)?)?.sum(D::Minus1)?.neg()?;
            // s = s ⊙ w[j] + v[i]·k[j] + sa[i]·b[j]
            let decayed = s.broadcast_mul(&w_t.unsqueeze(1)?)?;
            let vk = v_t.unsqueeze(2)?.broadcast_mul(&k_t.unsqueeze(1)?)?;
            let sb = sa.unsqueeze(2)?.broadcast_mul(&b_t.unsqueeze(1)?)?;
            s = decayed.add(&(vk.add(&sb)?))?;
            // out[h,i] = Σ_j s[h,i,j] · r[h,j]
            let o = s.broadcast_mul(&r_t.unsqueeze(1)?)?.sum(D::Minus1)?;
            out_rows.push(o);
        }
        self.state = s.contiguous()?;
        let out = Tensor::stack(&out_rows, 0)?; // [T, hc, hs]

        // Per-head LayerNorm (no affine) → affine with ln_w/ln_b.
        let gn = norm_no_affine(&out, GROUP_NORM_EPS)?
            .reshape((t_len, self.hidden))?
            .broadcast_mul(&self.ln_w.unsqueeze(0)?)?
            .broadcast_add(&self.ln_b.unsqueeze(0)?)?;

        // rk = Σ_d k·r·r_k per head → out += v ⊙ rk.
        let rk = k3
            .mul(&r3)?
            .broadcast_mul(&self.r_k.reshape((1, hc, hs))?)?
            .sum(D::Minus1)?; // [T, hc]
        let out = gn.broadcast_add(&v3.broadcast_mul(&rk.unsqueeze(2)?)?.reshape((t_len, self.hidden))?)?;

        // Optional gating, then the output projection.
        let out = if let (Some(g1), Some(g2)) = (&self.g1, &self.g2) {
            let xg = lerp(&self.lerp_fused[5], &sx, x)?;
            let g = g2.forward(&sigmoid(&g1.forward(&xg)?)?)?;
            out.broadcast_mul(&g)?
        } else {
            out
        };
        Ok(self.output.forward(&out)?)
    }
}

// ─────────────────────────── channel mix ────────────────────────────

struct ChannelMix {
    lerp_k: Tensor,
    key: QMatMul,
    value: QMatMul,
}

impl ChannelMix {
    fn forward(&self, x: &Tensor, x_prev: &Tensor) -> Result<Tensor> {
        let sx = x_prev.sub(x)?;
        let xk = x.broadcast_add(&sx.broadcast_mul(&self.lerp_k.unsqueeze(0)?)?)?;
        let k = self.key.forward(&xk)?.relu()?.powf(2.0)?;
        Ok(self.value.forward(&k)?)
    }
}

// ────────────────────────────── model ───────────────────────────────

struct Layer {
    attn_norm: LayerNorm,
    attn_norm_2: LayerNorm,
    time_mix: TimeMix,
    channel_mix: ChannelMix,
    /// Previous token's `attn_norm` (zeros before the first token).
    att_shift: Tensor,
    /// Previous token's `ffn_norm` (zeros before the first token).
    ffn_shift: Tensor,
}

pub struct Rwkv7Model {
    arch: String,
    embeddings: Arc<candle_core::quantized::QTensor>,
    output: QMatMul,
    tok_norm: LayerNorm,
    out_norm: LayerNorm,
    layers: Vec<Layer>,
}

impl crate::model::registry::DynModel for Rwkv7Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, _index_pos: usize) -> Result<Tensor> {
        let (b, t_len) = (input.dims()[0], input.dims()[1]);
        if b != 1 {
            return Err(Error::Model(format!("rwkv7: batch {b} > 1 not supported")));
        }
        let ids = input.squeeze(0)?;
        let mut x = self
            .embeddings
            .embedding(&ids)?
            .to_dtype(DType::F32)?;
        x = self.tok_norm.forward(&x)?;
        let mut v_first: Option<Tensor> = None;
        for layer in &mut self.layers {
            let att_norm = layer.attn_norm.forward(&x)?;
            let x_prev = Tensor::cat(
                &[layer.att_shift.clone(), att_norm.narrow(0, 0, t_len - 1)?],
                0,
            )?;
            let attn_out = layer.time_mix.forward(&att_norm, &x_prev, &mut v_first)?;
            x = x.add(&attn_out)?;

            let ffn_norm = layer.attn_norm_2.forward(&x)?;
            let x_prev2 = Tensor::cat(
                &[layer.ffn_shift.clone(), ffn_norm.narrow(0, 0, t_len - 1)?],
                0,
            )?;
            let ch_out = layer.channel_mix.forward(&ffn_norm, &x_prev2)?;
            x = x.add(&ch_out)?;

            layer.att_shift = att_norm.narrow(0, t_len - 1, 1)?.contiguous()?;
            layer.ffn_shift = ffn_norm.narrow(0, t_len - 1, 1)?.contiguous()?;
        }
        let x = self.out_norm.forward(&x)?;
        let last = x.narrow(0, t_len - 1, 1)?;
        let logits = self
            .output
            .forward(&last)?
            .to_dtype(DType::F32)?
            .squeeze(0)?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            // Shift states are `[1, hidden]` so they concatenate with the
            // `[T-1, hidden]` activations.
            let device = layer.att_shift.device().clone();
            layer.att_shift = Tensor::zeros((1, layer.att_shift.dims()[1]), DType::F32, &device)
                .unwrap_or_else(|e| {
                    tracing::warn!("rwkv7: shift reset failed: {e}");
                    layer.att_shift.clone()
                });
            let device = layer.ffn_shift.device().clone();
            layer.ffn_shift = Tensor::zeros((1, layer.ffn_shift.dims()[1]), DType::F32, &device)
                .unwrap_or_else(|e| {
                    tracing::warn!("rwkv7: shift reset failed: {e}");
                    layer.ffn_shift.clone()
                });
            if let Err(e) = layer.time_mix.reset() {
                tracing::warn!("rwkv7: state reset failed: {e}");
            }
        }
    }
}

/// Build an `rwkv7` model. See the module docs for the architecture.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn crate::model::registry::DynModel>> {
    let content = loaded.take_content().map_err(|e| {
        Error::Model(format!("rwkv7: GGUF content unavailable: {e}"))
    })?;
    let mut gg = Gguf::new(content, &mut loaded.file, device.clone());
    let meta = crate::model::common::Meta::new(gg.metadata(), "rwkv7");

    let n_layer = meta.req_u32("block_count")?;
    let hidden = meta.req_u32("embedding_length")?;
    let ffn = meta.req_u32("feed_forward_length")?;
    let head_size = meta.req_u32("wkv.head_size")?;
    let n_lora_decay = meta.req_u32("attention.decay_lora_rank")?;
    let n_lora_iclr = meta.req_u32("attention.iclr_lora_rank")?;
    let n_lora_vres = meta.req_u32("attention.value_residual_mix_lora_rank")?;
    // Gating is optional — its absence means no g stream at all.
    let n_lora_gate = meta.opt_u32("attention.gate_lora_rank", 0);
    let has_gating = n_lora_gate > 0;
    let eps = meta.opt_f32("attention.layer_norm_epsilon", 1e-5) as f64;
    if hidden % head_size != 0 {
        return Err(Error::Model(format!(
            "rwkv7: embedding_length {hidden} not divisible by head_size {head_size}"
        )));
    }
    let head_count = hidden / head_size;
    tracing::info!(
        layers = n_layer,
        hidden,
        ffn,
        head_count,
        head_size,
        gating = has_gating,
        "building rwkv7"
    );

    let tok_norm = load_ln(&mut gg, "token_embd_norm.weight", "token_embd_norm.bias", eps, device)?;
    let out_norm = load_ln(&mut gg, "output_norm.weight", "output_norm.bias", eps, device)?;
    let embeddings = std::sync::Arc::new(gg.tensor("token_embd.weight")?);
    let output = qm(&mut gg, "output.weight")?;

    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let p = format!("blk.{i}");
        let attn_norm =
            load_ln(&mut gg, &format!("{p}.attn_norm.weight"), &format!("{p}.attn_norm.bias"), eps, device)?;
        let attn_norm_2 = load_ln(
            &mut gg,
            &format!("{p}.attn_norm_2.weight"),
            &format!("{p}.attn_norm_2.bias"),
            eps,
            device,
        )?;

        // The fused lerp tensor: ggml shape {n_embd, 1, 1, 6} → candle
        // [6, 1, 1, n_embd] — one [hidden] slice per stream.
        let lf = gg
            .tensor(&format!("{p}.time_mix_lerp_fused.weight"))
            .map_err(|e| Error::Model(format!("rwkv7: missing lerp_fused: {e}")))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        if lf.dims() != [if has_gating { 6 } else { 5 }, 1, 1, hidden] {
            // llama.cpp pads to 6 lanes but converts always write 6; accept
            // either, indexing only the streams we use.
            tracing::warn!(
                "rwkv7: lerp_fused dims {:?} (expected [6,1,1,{hidden}])",
                lf.dims()
            );
        }
        let lanes = lf.dims()[0].max(if has_gating { 6 } else { 5 });
        let lerp_fused = (0..lanes)
            .map(|j| Ok(lf.narrow(0, j, 1)?.reshape((hidden,))?))
            .collect::<candle_core::Result<Vec<_>>>()?;

        let time_mix = TimeMix {
            hidden,
            head_count,
            head_size,
            key: qm(&mut gg, &format!("{p}.time_mix_key.weight"))?,
            value: qm(&mut gg, &format!("{p}.time_mix_value.weight"))?,
            receptance: qm(&mut gg, &format!("{p}.time_mix_receptance.weight"))?,
            output: qm(&mut gg, &format!("{p}.time_mix_output.weight"))?,
            w0: vec1(&mut gg, &format!("{p}.time_mix_w0.weight"), device)?,
            w1: qm(&mut gg, &format!("{p}.time_mix_w1.weight"))?,
            w2: qm(&mut gg, &format!("{p}.time_mix_w2.weight"))?,
            a0: vec1(&mut gg, &format!("{p}.time_mix_a0.weight"), device)?,
            a1: qm(&mut gg, &format!("{p}.time_mix_a1.weight"))?,
            a2: qm(&mut gg, &format!("{p}.time_mix_a2.weight"))?,
            v0: if i > 0 {
                Some(vec1(&mut gg, &format!("{p}.time_mix_v0.weight"), device)?)
            } else {
                None
            },
            v1: if i > 0 { Some(qm(&mut gg, &format!("{p}.time_mix_v1.weight"))?) } else { None },
            v2: if i > 0 { Some(qm(&mut gg, &format!("{p}.time_mix_v2.weight"))?) } else { None },
            g1: if has_gating { Some(qm(&mut gg, &format!("{p}.time_mix_g1.weight"))?) } else { None },
            g2: if has_gating { Some(qm(&mut gg, &format!("{p}.time_mix_g2.weight"))?) } else { None },
            lerp_fused,
            k_k: vec1(&mut gg, &format!("{p}.time_mix_k_k.weight"), device)?,
            k_a: vec1(&mut gg, &format!("{p}.time_mix_k_a.weight"), device)?,
            r_k: vec1(&mut gg, &format!("{p}.time_mix_r_k.weight"), device)?,
            ln_w: vec1(&mut gg, &format!("{p}.time_mix_ln.weight"), device)?,
            ln_b: vec1(&mut gg, &format!("{p}.time_mix_ln.bias"), device)?,
            state: Tensor::zeros((head_count, head_size, head_size), DType::F32, device)?,
        };

        let channel_mix = ChannelMix {
            lerp_k: vec1(&mut gg, &format!("{p}.channel_mix_lerp_k.weight"), device)?,
            key: qm(&mut gg, &format!("{p}.channel_mix_key.weight"))?,
            value: qm(&mut gg, &format!("{p}.channel_mix_value.weight"))?,
        };

        layers.push(Layer {
            attn_norm,
            attn_norm_2,
            time_mix,
            channel_mix,
            att_shift: Tensor::zeros((1, hidden), DType::F32, device)?,
            ffn_shift: Tensor::zeros((1, hidden), DType::F32, device)?,
        });
    }

    Ok(Box::new(Rwkv7Model {
        arch: "rwkv7".into(),
        embeddings,
        output,
        tok_norm,
        out_norm,
        layers,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WKV7 recurrence, hand-rolled scalar reference vs the tensor form
    /// used in `TimeMix::forward`: same math, same index roles as the ggml
    /// CPU kernel. Builds a minimal single-head "layer" around the raw
    /// recurrence only (weights bypassed — we drive the kernel contract).
    #[test]
    fn wkv7_recurrence_matches_scalar_reference() {
        let dev = Device::Cpu;
        let hs = 4usize; // head size (single head)
        let t_len = 7usize;

        // Random-ish inputs in a safe range.
        let mk = |seed: f32, n: usize| -> Vec<f32> {
            (0..n).map(|i| (((seed * 7.3 + i as f32 * 1.37).sin() * 0.9))).collect()
        };
        let r = mk(1.0, hs);
        let w = mk(2.0, hs).iter().map(|v| 0.5 + 0.5 * v.abs()).collect::<Vec<_>>();
        let k = mk(3.0, hs);
        let v = mk(4.0, hs);
        let kk = mk(5.0, hs).iter().map(|x| x / 2.0).collect::<Vec<_>>();
        let a = mk(6.0, hs).iter().map(|x| 0.5 + 0.25 * x).collect::<Vec<_>>();

        // Scalar reference — EXACTLY the ggml kernel loop (single head).
        let mut s_ref = vec![vec![0f32; hs]; hs];
        let mut out_ref = vec![0f32; hs];
        // sa_i = -Σ_j kk[j]·s[i][j]
        let sa: Vec<f32> = (0..hs)
            .map(|i| -(0..hs).map(|j| kk[j] * s_ref[i][j]).sum::<f32>())
            .collect();
        for i in 0..hs {
            for j in 0..hs {
                s_ref[i][j] = s_ref[i][j] * w[j] + v[i] * k[j] + sa[i] * (kk[j] * a[j]);
                out_ref[i] += s_ref[i][j] * r[j];
            }
        }

        // Tensor form (one token, one head) — the ops from TimeMix::forward.
        let r_t = Tensor::from_vec(r.clone(), (1, hs), &dev).unwrap();
        let w_t = Tensor::from_vec(w.clone(), (1, hs), &dev).unwrap();
        let k_t = Tensor::from_vec(k.clone(), (1, hs), &dev).unwrap();
        let v_t = Tensor::from_vec(v.clone(), (1, hs), &dev).unwrap();
        let kk_t = Tensor::from_vec(kk.clone(), (1, hs), &dev).unwrap();
        let a_t = Tensor::from_vec(a.clone(), (1, hs), &dev).unwrap();
        let b_t = kk_t.broadcast_mul(&a_t).unwrap();
        let s = Tensor::zeros((1, hs, hs), DType::F32, &dev).unwrap();

        let sa_t = s.broadcast_mul(&kk_t.unsqueeze(1).unwrap()).unwrap().sum(D::Minus1).unwrap().neg().unwrap();
        let decayed = s.broadcast_mul(&w_t.unsqueeze(1).unwrap()).unwrap();
        let vk = v_t.unsqueeze(2).unwrap().broadcast_mul(&k_t.unsqueeze(1).unwrap()).unwrap();
        let sb = sa_t.unsqueeze(2).unwrap().broadcast_mul(&b_t.unsqueeze(1).unwrap()).unwrap();
        let s2 = decayed.add(&(vk.add(&sb).unwrap())).unwrap();
        let out_t = s2.broadcast_mul(&r_t.unsqueeze(1).unwrap()).unwrap().sum(D::Minus1).unwrap();

        let out_v = out_t.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
        let s_v = s2.to_vec3::<f32>().unwrap();
        for i in 0..hs {
            assert!((out_v[i] - out_ref[i]).abs() < 1e-5, "out[{i}]: {} vs {}", out_v[i], out_ref[i]);
            for j in 0..hs {
                assert!(
                    (s_v[0][i][j] - s_ref[i][j]).abs() < 1e-5,
                    "s[{i}][{j}]: {} vs {}",
                    s_v[0][i][j],
                    s_ref[i][j]
                );
            }
        }
    }

    /// The channel mix contract: relu²  FFN with token-shift lerp.
    #[test]
    fn channel_mix_relu_squared() {
        let dev = Device::Cpu;
        // A "weight" of ones [out=2, in=3] f32 → plain QMatMul-free check via
        // manual math on the same formula.
        let x = Tensor::from_vec(vec![1.0f32, -2.0, 3.0], (1, 3), &dev).unwrap();
        let x_prev = Tensor::from_vec(vec![0.5f32, 0.5, 0.5], (1, 3), &dev).unwrap();
        let sx = x_prev.sub(&x).unwrap();
        let lerp_k = Tensor::from_vec(vec![0.25f32; 3], (3,), &dev).unwrap();
        let xk = x
            .broadcast_add(&sx.broadcast_mul(&lerp_k.unsqueeze(0).unwrap()).unwrap())
            .unwrap();
        // xk = [1 + (0.5-1)*0.25, -2 + (0.5+2)*0.25, 3 + (0.5-3)*0.25]
        //    = [0.875, -1.375, 2.375]
        let v: Vec<f32> = xk.to_vec2().unwrap()[0].clone();
        assert!((v[0] - 0.875).abs() < 1e-6);
        assert!((v[1] - (-1.375)).abs() < 1e-6);
        assert!((v[2] - 2.375).abs() < 1e-6);
        // relu²: [0.765625, 0, 5.640625]
        let r = xk.relu().unwrap().powf(2.0).unwrap();
        let r: Vec<f32> = r.to_vec2().unwrap()[0].clone();
        assert!((r[0] - 0.875f32 * 0.875).abs() < 1e-6);
        assert_eq!(r[1], 0.0);
        assert!((r[2] - 2.375f32 * 2.375).abs() < 1e-5);
    }

    /// Multi-token recurrence: state must compound across tokens (token 2's
    /// output depends on token 1's write).
    #[test]
    fn recurrence_compounds_across_tokens() {
        let dev = Device::Cpu;
        let hs = 3usize;
        let mk = |seed: f32| -> Vec<f32> {
            (0..hs).map(|i| ((seed + i as f32 * 1.7).sin() * 0.8)).collect()
        };
        let w: Vec<f32> = mk(1.0).iter().map(|v| 0.6 + 0.3 * v.abs()).collect();
        let k = mk(2.0);
        let v = mk(3.0);
        let kk: Vec<f32> = mk(4.0).iter().map(|x| x * 0.3).collect();
        let a: Vec<f32> = mk(5.0).iter().map(|x| 0.5 + 0.2 * x).collect();
        let r1 = mk(6.0);
        let r2 = mk(7.0);

        let run = |r_seq: &[&[f32]]| -> Vec<f32> {
            let mut s = vec![vec![0f32; hs]; hs];
            let mut outs = Vec::new();
            for r in r_seq {
                let sa: Vec<f32> = (0..hs)
                    .map(|i| -(0..hs).map(|j| kk[j] * s[i][j]).sum::<f32>())
                    .collect();
                let mut o = vec![0f32; hs];
                for i in 0..hs {
                    for j in 0..hs {
                        s[i][j] = s[i][j] * w[j] + v[i] * k[j] + sa[i] * (kk[j] * a[j]);
                        o[i] += s[i][j] * r[j];
                    }
                }
                outs.extend(o);
            }
            outs
        };
        let both = run(&[&r1, &r2]);
        let only2 = run(&[&r2]);
        // The second token's output must differ when preceded by token 1 —
        // the state carried over.
        for i in 0..hs {
            assert!(
                (both[hs + i] - only2[i]).abs() > 1e-6,
                "token-2 output must depend on token-1 state"
            );
        }
    }
}
