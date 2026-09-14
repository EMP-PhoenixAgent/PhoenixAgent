//! Qwen 3.5 — the hybrid gated-delta-net family (`qwen35`).
//!
//! Decoder layers come in two kinds, routed by `attention.recurrent_layers`
//! (bool array) or the `full_attention_interval` fallback (every Nth layer is
//! full attention — layers 3, 7, 11, … when N=4, so 3 of 4 layers are
//! recurrent):
//!
//! - **Recurrent (GDN)** layers — a *gated delta net* linear attention:
//!   fused `attn_qkv` + separate `attn_gate` (z), a short causal depthwise
//!   conv (`ssm_conv1d`, kernel 4) with silu, L2-normalized per-head q/k,
//!   per-head scalar decay `g = softplus(ssm_alpha(x) + ssm_dt_bias) ⊙ ssm_a`
//!   (the stored `ssm_a` is already `-exp(A_log)`), `beta = sigmoid(ssm_beta(x))`
//!   and the state recurrence `S ← exp(g)·S + beta·k⊗v`, read as `o = S·q`.
//!   Output runs through a per-head gated RMSNorm (`ssm_norm ⊙ silu(z)`) and
//!   `ssm_out`.
//! - **Full attention** layers — standard GQA, but `attn_q` is a **fused
//!   Q+gate** projection (`2·head_dim` per head, interleaved `[q | gate]`)
//!   whose output is scaled by `sigmoid(gate)`; q/k are RMS-normed per head
//!   and carry a partial rope (first `rope.dimension_count` of the 256-dim
//!   heads, base 1e7; the MRoPE `rope.dimension_sections` degenerate to one
//!   contiguous rope for text-only models — sections with equal positions).
//!
//! Prefill of the recurrent layers is **chunked** (Mamba2-style SSD): per
//! chunk of `CHUNK` tokens, the decay is the `segsum`-matrix
//! `D[i,j] = exp(G_i − G_j)` of the cumulative log-decay `G = cumsum(g)`, the
//! output is `((Q·Kᵀ) ⊙ D ⊙ β)·V + exp(G) ⊙ (Q·S_in)`, and the state updates
//! as `S ← exp(G_L)·S_in + (Kᵀ ⊙ (exp(G_L − G)·β))·V`. Decode is the same
//! code with a 1-token chunk — one path, no separate stepwise implementation.
//!
//! FFN is dense SwiGLU on all current Qwen 3.5 releases; `qwen35moe`-style
//! `ffn_gate_inp` is rejected cleanly until a real GGUF pins its names.
//! MTP/NextN blocks (`nextn.*`) are speculative-decode only — detected,
//! warned, ignored. KV storage: one [`KvSlot`] per **full** layer (the
//! recurrent layers hold conv + SSM state instead), so the KV footprint is
//! 1/interval of a dense model's.
//!
//! Verified against a real `Qwen3.5-0.8B-Q8_0` GGUF (unsloth): hparams
//! (GQA 8/2 heads × 256, rope 64 dims, conv 4, state 128, groups 16,
//! dt-rank 16, inner 2048, interval 4) and every tensor name/shape.

use crate::error::{Error, Result};
use crate::model::common::{causal_mask, scaled_inv_freq, KvSlot, Meta, RopeTables, Scaling};
use crate::model::gguf::{LoadedModel, ModelFile};
use crate::model::registry::DynModel;
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;
use candle_transformers::models::quantized_qwen3::Gguf;
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

/// Prefill chunk length for the delta-net recurrence. 64 keeps the
/// (heads, L, L) decay matrices small while amortizing the matmuls.
const CHUNK: usize = 64;


// ───────────────────────── full attention layers ─────────────────────────────

/// One full-attention layer: GQA with a fused Q+gate projection.
struct FullAttention {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    rope: std::sync::Arc<RopeTables>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_dims: usize,
    kv: KvSlot,
}

impl FullAttention {
    fn new<R: std::io::Seek + std::io::Read>(
        gg: &mut Gguf<R>,
        prefix: &str,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        rope_dims: usize,
        eps: f64,
        rope: std::sync::Arc<RopeTables>,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            // (2·head_dim·n_head, hidden) — [q | gate] interleaved per head.
            wq: gg.qmatmul(&format!("{prefix}.attn_q.weight"))?,
            wk: gg.qmatmul(&format!("{prefix}.attn_k.weight"))?,
            wv: gg.qmatmul(&format!("{prefix}.attn_v.weight"))?,
            wo: gg.qmatmul(&format!("{prefix}.attn_output.weight"))?,
            q_norm: gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), eps)?,
            k_norm: gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), eps)?,
            rope,
            n_head,
            n_kv_head,
            head_dim,
            rope_dims,
            kv: KvSlot::default(),
        })
    }

    fn forward(&mut self, x: &Tensor, offset: usize) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let dtype = x.dtype();
        let device = x.device();

        let qg = self.wq.forward(x)?.reshape((b * seq_len, self.n_head, 2 * self.head_dim))?;
        // Per-head interleave: head h owns [h·2hd, h·2hd+2hd) — q first, gate second.
        let q = qg
            .narrow(2, 0, self.head_dim)?
            .reshape((b, seq_len, self.n_head, self.head_dim))?
            .contiguous()?;
        let gate = qg
            .narrow(2, self.head_dim, self.head_dim)?
            .reshape((b * seq_len, self.n_head, self.head_dim))?
            .contiguous()?;

        let k = self
            .wk
            .forward(x)?
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?;
        let v = self
            .wv
            .forward(x)?
            .reshape((b, seq_len, self.n_kv_head, self.head_dim))?;

        // Norms before rope, per head.
        let q = self.q_norm.forward(&q)?.transpose(1, 2)?.contiguous()?;
        let k = self.k_norm.forward(&k)?.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        // Partial rope: rotate the first `rope_dims`, pass the rest through.
        // Both halves are materialized and the concatenation is forced
        // contiguous: at decode (seq == 1) the narrowed `pass` view enters
        // `cat` with extent-1 dims, and candle's CUDA cat leaves the result
        // with head-interleaved strides — CPU matmuls copy such views, but
        // CUDA either rejects them or (older builds) reads them as garbage
        // and the NaN logits surface at sampling.
        let apply_rope = |t: &Tensor| -> candle_core::Result<Tensor> {
            if self.rope_dims < self.head_dim {
                let rot = t.narrow(3, 0, self.rope_dims)?.contiguous()?;
                let pass = t
                    .narrow(3, self.rope_dims, self.head_dim - self.rope_dims)?
                    .contiguous()?;
                let rot = self.rope.apply_half(&rot, dtype, offset)?;
                Tensor::cat(&[rot, pass], 3)?.contiguous()
            } else {
                self.rope.apply_half(t, dtype, offset)
            }
        };
        let q = apply_rope(&q)?;
        let k = apply_rope(&k)?;

        let (k, v) = self.kv.append(&k, &v)?;
        let k = repeat_kv(k, self.n_head / self.n_kv_head)?.contiguous()?;
        let v = repeat_kv(v, self.n_head / self.n_kv_head)?.contiguous()?;

        let kq_scale = 1f64 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * kq_scale)?;
        let mask = causal_mask(device, dtype, b, seq_len, offset, None)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .reshape((b * seq_len, self.n_head, self.head_dim))?;

        // Output gate: attention result × sigmoid(per-head gate).
        let gated = ctx.broadcast_mul(&candle_nn::ops::sigmoid(&gate)?)?;
        let gated = gated.reshape((b, seq_len, self.n_head * self.head_dim))?.contiguous()?;
        self.wo.forward(&gated.to_dtype(dtype)?)
    }
}

// ─────────────────────── recurrent (GDN) layers ──────────────────────────────

/// One recurrent gated-delta-net layer with its conv + SSM state.
struct GdnAttn {
    qkv: QMatMul,
    gate_proj: QMatMul,
    beta_w: QMatMul,
    alpha_w: QMatMul,
    out_proj: QMatMul,
    head_norm: RmsNorm,
    /// `(conv_dim, d_conv)` F32 — dequantized (tiny).
    conv_w: Tensor,
    /// `(n_v_heads,)` F32 bias added to alpha before softplus.
    dt_bias: Tensor,
    /// `(n_v_heads,)` F32, already `-exp(A_log)`.
    a: Tensor,
    d_conv: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    d_inner: usize,
    eps: f32,
    /// GGUF weights carry llama.cpp's channel order (v/z permuted so the
    /// k→v head repeat TILES); safetensors carry HF's (repeat_interleave
    /// BLOCKS). Equal head counts make the two coincide.
    kv_repeat_tiled: bool,
    /// `(d_conv − 1, conv_dim)` F32 — the last inputs seen.
    conv_state: Tensor,
    /// `(n_v_heads, head_k_dim, head_v_dim)` F32 — the delta-rule state.
    ssm_state: Tensor,
}

impl GdnAttn {
    #[allow(clippy::too_many_arguments)]
    fn new<R: std::io::Seek + std::io::Read>(
        gg: &mut Gguf<R>,
        prefix: &str,
        d_conv: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        d_inner: usize,
        eps: f64,
        kv_repeat_tiled: bool,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let conv_dim = d_inner + 2 * n_k_heads * head_k_dim;
        let conv_w = gg
            .tensor(&format!("{prefix}.ssm_conv1d.weight"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        if conv_w.dims() != &[conv_dim, d_conv] {
            candle_core::bail!(
                "{prefix}.ssm_conv1d.weight is {:?}, expected ({conv_dim}, {d_conv})",
                conv_w.dims()
            );
        }
        let dt_bias = gg
            .tensor(&format!("{prefix}.ssm_dt.bias"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let a = gg
            .tensor(&format!("{prefix}.ssm_a"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        let conv_state = Tensor::zeros((d_conv - 1, conv_dim), DType::F32, device)?;
        let ssm_state = Tensor::zeros((n_v_heads, head_k_dim, head_v_dim), DType::F32, device)?;
        Ok(Self {
            qkv: gg.qmatmul(&format!("{prefix}.attn_qkv.weight"))?,
            gate_proj: gg.qmatmul(&format!("{prefix}.attn_gate.weight"))?,
            beta_w: gg.qmatmul(&format!("{prefix}.ssm_beta.weight"))?,
            alpha_w: gg.qmatmul(&format!("{prefix}.ssm_alpha.weight"))?,
            out_proj: gg.qmatmul(&format!("{prefix}.ssm_out.weight"))?,
            head_norm: gg.rms_norm(&format!("{prefix}.ssm_norm.weight"), eps)?,
            conv_w,
            dt_bias,
            a,
            d_conv,
            n_k_heads,
            n_v_heads,
            head_k_dim,
            head_v_dim,
            d_inner,
            eps: eps as f32,
            kv_repeat_tiled,
            conv_state,
            ssm_state,
        })
    }

    fn reset(&mut self) -> candle_core::Result<()> {
        let device = self.conv_state.device().clone();
        let conv_dims = self.conv_state.dims().to_vec();
        let ssm_dims = self.ssm_state.dims().to_vec();
        self.conv_state = Tensor::zeros(conv_dims, DType::F32, &device)?;
        self.ssm_state = Tensor::zeros(ssm_dims, DType::F32, &device)?;
        Ok(())
    }

    fn forward(&mut self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, seq_len, _hidden) = x.dims3()?;
        if b != 1 {
            candle_core::bail!("qwen35 GDN layers support batch 1 (got {b})");
        }
        let device = x.device();
        let conv_dim = self.d_inner + 2 * self.n_k_heads * self.head_k_dim;

        // Per-token scalars from the *pre-conv* layer input.
        let beta = candle_nn::ops::sigmoid(&self.beta_w.forward(x)?)?; // (1, t, n_v)
        let alpha = self.alpha_w.forward(x)?; // (1, t, n_v)
        let alpha = alpha.squeeze(0)?.broadcast_add(&self.dt_bias)?; // (t, n_v)
        let gate = softplus(&alpha)?.broadcast_mul(&self.a)?; // (t, n_v)
        // (n_v, t, 1) for the chunk math.
        let beta = beta
            .squeeze(0)?
            .transpose(0, 1)?
            .reshape((self.n_v_heads, seq_len, 1))?
            .contiguous()?;
        let gate = gate
            .transpose(0, 1)?
            .reshape((self.n_v_heads, seq_len, 1))?
            .contiguous()?;

        let qkv = self.qkv.forward(x)?.squeeze(0)?.to_dtype(DType::F32)?; // (t, conv_dim)
        let z = self.gate_proj.forward(x)?.squeeze(0)?; // (t, d_inner)

        // Causal depthwise conv over time, left-padded with the state.
        let padded = Tensor::cat(&[self.conv_state.clone(), qkv], 0)?.contiguous()?;
        let mut conv_out: Option<Tensor> = None;
        for j in 0..self.d_conv {
            // out[t] += w[:, j] ⊙ padded[t + j, :]
            let wj = self.conv_w.narrow(1, j, 1)?.squeeze(1)?; // (conv_dim,)
            let wj = wj.reshape((1, 1, conv_dim))?;
            let xs = padded
                .narrow(0, j, seq_len)?
                .reshape((1, seq_len, conv_dim))?;
            let term = wj.broadcast_mul(&xs)?;
            conv_out = Some(match conv_out {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        let conv_out = silu(&conv_out.expect("d_conv >= 1"))?; // (1, t, conv_dim)

        // Split [q | k | v] out of the convolved mix.
        let k_total = self.n_k_heads * self.head_k_dim;
        let v_total = self.n_v_heads * self.head_v_dim;
        let q_c = conv_out.narrow(2, 0, k_total)?;
        let k_c = conv_out.narrow(2, k_total, k_total)?;
        let v_c = conv_out.narrow(2, 2 * k_total, v_total)?;

        // Per-head L2 normalization of q and k (eps 1e-6, matching the FLA
        // kernel), then the kernel's fixed query scaling by d_k^-1/2.
        let q_c = l2_norm(
            &q_c.reshape((seq_len, self.n_k_heads, self.head_k_dim))?.contiguous()?,
            1e-6,
        )?;
        let k_c = l2_norm(
            &k_c.reshape((seq_len, self.n_k_heads, self.head_k_dim))?.contiguous()?,
            1e-6,
        )?;
        let q_scale = 1f32 / (self.head_k_dim as f32).sqrt();
        let v_c =
            v_c.reshape((seq_len, self.n_v_heads, self.head_v_dim))?.contiguous()?;

        // (heads, t, dim) for batched chunk math; grow k heads to v heads if
        // the group counts differ (equal on every real Qwen 3.5 so far).
        let mut q_h = q_c.transpose(0, 1)?.contiguous()?;
        let mut k_h = k_c.transpose(0, 1)?.contiguous()?;
        if self.n_k_heads != self.n_v_heads {
            if self.n_v_heads % self.n_k_heads != 0 {
                candle_core::bail!(
                    "n_v_heads ({}) not a multiple of n_k_heads ({})",
                    self.n_v_heads,
                    self.n_k_heads
                );
            }
            let rep = self.n_v_heads / self.n_k_heads;
            // Duplicate each k head across its `rep` value heads — in the
            // layout the WEIGHT SOURCE expects (see ACRoad §6b rule 5):
            // GGUF converters permute v/z channels for llama.cpp's TILED
            // ggml_repeat ([k0..kN, k0..kN]); raw HF safetensors order pairs
            // them via transformers' repeat_interleave BLOCK ([k0, k0, k1,
            // k1]). The rep axis goes outside (tiled) or inside (block) the
            // head axis before the contiguous flatten — putting it between
            // head and time interleaves it with TIME and scrambles q/k.
            let grow = |t: &Tensor| -> candle_core::Result<Tensor> {
                let (_, t_len, d) = t.dims3()?;
                let (outer, inner) = if self.kv_repeat_tiled {
                    ((1, self.n_k_heads), (rep, self.n_k_heads))
                } else {
                    ((self.n_k_heads, 1), (self.n_k_heads, rep))
                };
                t.reshape((outer.0, outer.1, t_len, d))?
                    .broadcast_as((inner.0, inner.1, t_len, d))?
                    .contiguous()?
                    .reshape((self.n_v_heads, t_len, d))
            };
            q_h = grow(&q_h)?;
            k_h = grow(&k_h)?;
        }
        let v_h = v_c.transpose(0, 1)?.contiguous()?; // (n_v, t, d_v)
        let q_h = q_h.affine(q_scale as f64, 0.0)?;

        // Gated DELTA rule (not an additive SSM): per step, the state is
        // decayed, then updated with the CORRECTION (v − S·k)·β — the model
        // overwrites associations instead of accumulating them. Prefill uses
        // the chunked UT-transform kernel (WY representation) from the FLA /
        // transformers reference; decode (and the equivalence test) uses the
        // stepwise form.
        // Chunked UT-transform prefill (guarded) — see chunked_delta_rule.
        let (o, s) = chunked_delta_rule(&q_h, &k_h, &v_h, &beta, &gate, &self.ssm_state, CHUNK)?;
        self.ssm_state = s.contiguous()?;

        // Keep the last d_conv − 1 inputs as the new conv state.
        self.conv_state = padded.narrow(0, seq_len, self.d_conv - 1)?.contiguous()?;

        // Per-head gated RMSNorm: rms_norm(o) ⊙ silu(z), then the output proj.
        let o = o.transpose(0, 1)?.contiguous()?; // (t, n_v, d_v)
        let z3 = z.reshape((seq_len, self.n_v_heads, self.head_v_dim))?;
        let normed = self.head_norm.forward(&o)?;
        let gated = normed.broadcast_mul(&silu(&z3)?)?;
        let flat = gated.reshape((1, seq_len, self.d_inner))?;
        self.out_proj.forward(&flat.to_dtype(x.dtype())?)
    }
}

/// One stepwise (token-recurrent) delta-rule update — the decode kernel, and
/// the unconditionally-stable fallback for chunks whose WY solve is
/// ill-conditioned (see [`chunked_delta_rule`]). Tensors are `(h, 1, ·)`.
fn step_delta(
    s: &mut Tensor,
    outs: &mut Vec<Tensor>,
    q1: &Tensor,
    k1: &Tensor,
    v1: &Tensor,
    b1: &Tensor,
    g1: &Tensor,
    h: usize,
) -> candle_core::Result<()> {
    // S ← exp(g)·S; δ = (v − Sᵀk)·β; S += k⊗δ; o = Sᵀq.
    let k_t = k1.transpose(1, 2)?; // (h, d_k, 1)
    let v_t = v1.transpose(1, 2)?; // (h, d_v, 1)
    let decay = g1.exp()?; // (h, 1, 1)
    let s_decayed = s.broadcast_mul(&decay)?;
    let kv_mem = s_decayed.transpose(1, 2)?.matmul(&k_t)?; // (h, d_v, 1) = Sᵀk
    let delta = v_t
        .broadcast_sub(&kv_mem)?
        .broadcast_mul(&b1.reshape((h, 1, 1))?)?;
    let outer = k_t.matmul(&delta.transpose(1, 2)?)?; // (h, d_k, d_v)
    *s = s_decayed.add(&outer)?;
    let o = s.transpose(1, 2)?.matmul(&q1.transpose(1, 2)?)?; // (h, d_v, 1)
    outs.push(o.transpose(1, 2)?); // (h, 1, d_v)
    Ok(())
}

/// The chunked UT-transform gated delta rule over `(heads, t, ·)` tensors —
/// the model's prefill kernel, factored out so the equivalence test can drive
/// it with an arbitrary chunk size. Returns `(o, final_state)`.
///
/// **Numerical guard.** The WY solve `(I + M)⁻¹ = Σ(−M)^k` is exact in real
/// arithmetic (M is nilpotent), but when a chunk's keys are strongly
/// correlated the series entries — and therefore `v_new = (I+M)⁻¹v_β − …` —
/// reach magnitudes f32 cancellation cannot survive, and the state compounds
/// chunk-over-chunk until the logits NaN (real Qwen3.5-4B hits this from
/// ~400-token prefills up; the 0.8B and short prompts never did — CUDA
/// surfaced it first, CPU silently kept going a bit longer). When the solve's
/// entries exceed `INV_GUARD`, or a chunked update still yields non-finite
/// values, that chunk is redone with the token-recurrent form, which is
/// mathematically identical and unconditionally stable (it is what decode and
/// llama.cpp use for this architecture). Well-conditioned chunks keep the
/// fast path — the guard only fires on the pathological ones.
fn chunked_delta_rule(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    gate: &Tensor,
    s0: &Tensor,
    chunk: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    /// Beyond this, (I+M)⁻¹ entries destroy more precision than the delta
    /// rule's cancellation can afford in f32.
    const INV_GUARD: f32 = 1e4;
    chunked_delta_rule_guarded(q, k, v, beta, gate, s0, chunk, INV_GUARD)
}

/// [`chunked_delta_rule`] with the guard threshold exposed for tests (a tiny
/// threshold forces the stepwise fallback on, proving it is exact).
fn chunked_delta_rule_guarded(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    gate: &Tensor,
    s0: &Tensor,
    chunk: usize,
    inv_guard: f32,
) -> candle_core::Result<(Tensor, Tensor)> {
    let device = q.device();
    let (h, t, _dk) = q.dims3()?;
    let mut s = s0.clone();
    let mut outs = Vec::new();
    let mut start = 0usize;
    while start < t {
        let len = chunk.min(t - start);
        let qc = q.narrow(1, start, len)?;
        let kc = k.narrow(1, start, len)?;
        let vc = v.narrow(1, start, len)?;
        let bc = beta.narrow(1, start, len)?;
        let gc = gate.narrow(1, start, len)?;
        if len == 1 {
            step_delta(&mut s, &mut outs, &qc, &kc, &vc, &bc, &gc, h)?;
            start += 1;
            continue;
        }

        let stepwise_fallback = |s: &mut Tensor,
                                 outs: &mut Vec<Tensor>|
         -> candle_core::Result<()> {
            for pos in 0..len {
                step_delta(
                    s,
                    outs,
                    &qc.narrow(1, pos, 1)?,
                    &kc.narrow(1, pos, 1)?,
                    &vc.narrow(1, pos, 1)?,
                    &bc.narrow(1, pos, 1)?,
                    &gc.narrow(1, pos, 1)?,
                    h,
                )?;
            }
            Ok(())
        };

        let g_cum = gc.cumsum(1)?;
        let g_last = g_cum.narrow(1, len - 1, 1)?;
        let causal = causal_f32(device, len)?;
        let g_rows = g_cum.broadcast_as((h, len, len))?;
        let d_mat = g_rows
            .broadcast_sub(&g_cum.transpose(1, 2)?)?
            .broadcast_mul(&causal)?
            .exp()?
            .broadcast_mul(&causal)?;
        let k_beta = kc.broadcast_mul(&bc)?;
        let v_beta = vc.broadcast_mul(&bc)?;
        let ut_raw = k_beta.matmul(&kc.transpose(1, 2)?)?.broadcast_mul(&d_mat)?;
        let strict_mask = strict_lower(device, len)?;
        let m_tri = ut_raw.broadcast_mul(&strict_mask)?;
        let neg_m = m_tri.affine(-1.0, 0.0)?;
        let eye = Tensor::eye(len, DType::F32, device)?
            .broadcast_as((h, len, len))?
            .contiguous()?;
        let mut t_series = eye;
        let mut p_k = neg_m;
        let mut m_len = 1usize;
        while m_len < len {
            t_series = t_series.broadcast_matmul(&p_k)?.broadcast_add(&t_series)?;
            p_k = p_k.broadcast_matmul(&p_k)?;
            m_len *= 2;
        }

        let inv_max = t_series
            .abs()?
            .flatten_all()?
            .max_all()?
            .to_scalar::<f32>()?;
        if !inv_max.is_finite() || inv_max > inv_guard {
            // Ill-conditioned WY solve — token-recurrent form for this chunk.
            stepwise_fallback(&mut s, &mut outs)?;
            start += len;
            continue;
        }

        let new_values = t_series.matmul(&v_beta)?;
        let decayed_k_beta = k_beta.broadcast_mul(&g_cum.exp()?)?;
        let k_cumdecay = t_series.matmul(&decayed_k_beta)?;
        let intra_att = qc.matmul(&kc.transpose(1, 2)?)?.broadcast_mul(&d_mat)?;
        let q_scaled = qc.broadcast_mul(&g_cum.exp()?)?;
        let k_out = kc.broadcast_mul(&g_last.broadcast_sub(&g_cum)?.exp()?)?;
        let chunk_decay = g_last.exp()?;
        let s_in = s.clone();
        let v_new = new_values.broadcast_sub(&k_cumdecay.matmul(&s)?)?;
        let inter = q_scaled.matmul(&s)?;
        let o = inter.add(&intra_att.matmul(&v_new)?)?;
        let s_next = s
            .broadcast_mul(&chunk_decay)?
            .add(&k_out.transpose(1, 2)?.matmul(&v_new)?)?;

        // Belt and braces: even a "passing" solve can lose the cancellation —
        // if anything came out non-finite, redo the chunk stepwise.
        let ok = |t: &Tensor| {
            t.mul(t)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.sum_all())
                .and_then(|t| t.to_scalar::<f32>())
                .map(|v| v.is_finite())
                .unwrap_or(false)
        };
        if ok(&o) && ok(&s_next) {
            outs.push(o);
            s = s_next;
        } else {
            s = s_in;
            stepwise_fallback(&mut s, &mut outs)?;
        }
        start += len;
    }
    Ok((Tensor::cat(&outs, 1)?, s))
}

/// `silu(x) = x · sigmoid(x)`.
fn silu(x: &Tensor) -> candle_core::Result<Tensor> {
    let sig = candle_nn::ops::sigmoid(x)?;
    x.broadcast_mul(&sig)
}

/// `softplus(x) = ln(1 + eˣ)` (F32-stable form matching candle's Mamba).
fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    (x.exp()? + 1.0)?.log()
}

/// L2-normalize the last dim: `x / √(Σx² + eps)` (per-head q/k normalization).
fn l2_norm(x: &Tensor, eps: f32) -> candle_core::Result<Tensor> {
    let last = x.dims()[x.dims().len() - 1];
    let norm = x
        .broadcast_mul(x)?
        .sum_keepdim(candle_core::D::Minus1)?
        .broadcast_add(&Tensor::new(eps, x.device())?)?
        .sqrt()?;
    let _ = last;
    x.broadcast_div(&norm)
}

/// Lower-triangular ones `(L, L)` — entry [i, j] = 1 iff j ≤ i (the identity
/// cumsummed DOWN the rows: column j's 1 fills every row i ≥ j).
fn causal_f32(device: &Device, l: usize) -> candle_core::Result<Tensor> {
    Ok(Tensor::eye(l, DType::F32, device)?.cumsum(0)?)
}

/// STRICT lower-triangular ones `(L, L)` — entry [i, j] = 1 iff j < i (causal
/// minus the diagonal).
fn strict_lower(device: &Device, l: usize) -> candle_core::Result<Tensor> {
    Ok((causal_f32(device, l)? - Tensor::eye(l, DType::F32, device)?)?)
}

// ──────────────────────────────── the model ──────────────────────────────────

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
        self.down.forward(&silu(&gate)?.broadcast_mul(&up)?)
    }
}

enum LayerKind {
    Full(FullAttention),
    Recurrent(GdnAttn),
}

struct Layer {
    attn_norm: RmsNorm,
    post_norm: RmsNorm,
    kind: LayerKind,
    ffn: Mlp,
}

pub struct Qwen35Model {
    arch: String,
    /// Quantized embedding table — rows fetched lazily on lookup.
    embeddings: std::sync::Arc<candle_core::quantized::QTensor>,
    /// Tied output head (token_embd as QMatMul).
    output: QMatMul,
    layers: Vec<Layer>,
    norm: RmsNorm,
    dtype: DType,
}

impl DynModel for Qwen35Model {
    fn arch(&self) -> &str {
        &self.arch
    }

    fn forward(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = self.forward_inner(input, index_pos)?;
        Ok(logits)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            match &mut layer.kind {
                LayerKind::Full(attn) => attn.kv.reset(),
                LayerKind::Recurrent(gdn) => {
                    if let Err(e) = gdn.reset() {
                        tracing::warn!("qwen35 state reset failed: {e}");
                    }
                }
            }
        }
    }
}

impl Qwen35Model {
    fn forward_inner(&mut self, input: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b, seq_len) = (input.dims()[0], input.dims()[1]);
        let mut x = self.embeddings.embedding(input)?.to_dtype(self.dtype)?;
        for layer in &mut self.layers {
            let residual = x.clone();
            let normed = layer.attn_norm.forward(&x)?;
            let attn_out = match &mut layer.kind {
                LayerKind::Full(attn) => attn.forward(&normed, index_pos)?,
                LayerKind::Recurrent(gdn) => gdn.forward(&normed)?,
            };
            let x2 = (residual + attn_out)?;
            let ffn_in = layer.post_norm.forward(&x2)?;
            let ffn_out = layer.ffn.forward(&ffn_in)?;
            x = (x2 + ffn_out)?;
        }
        let x = self.norm.forward(&x)?;
        // Last-position logits via the tied head.
        let last = x.narrow(1, seq_len - 1, 1)?;
        let logits = self
            .output
            .forward(&last)?
            .to_dtype(DType::F32)?
            .squeeze(1)?;
        Ok(logits)
    }
}

/// Build a `qwen35` model. See the module docs for the architecture.
pub fn build(loaded: &mut LoadedModel, device: &Device) -> Result<Box<dyn DynModel>> {
    let content = loaded.take_content().map_err(|e| {
        Error::Model(format!("qwen35: GGUF content unavailable: {e}"))
    })?;
    // GGUF weights follow llama.cpp's channel order; safetensors follow HF's
    // (see the kv_repeat_tiled doc).
    let kv_repeat_tiled = matches!(loaded.file, ModelFile::Gguf(_));
    let mut gg = Gguf::new(content, &mut loaded.file, device.clone());
    let meta = Meta::new(gg.metadata(), "qwen35");

    let n_layer = meta.req_u32("block_count")?;
    let hidden = meta.req_u32("embedding_length")?;
    let n_head = meta.opt_u32("attention.head_count", 32);
    let n_kv_head = meta.opt_u32("attention.head_count_kv", n_head);
    let head_dim = meta.opt_u32("attention.key_length", hidden / n_head.max(1));
    let rope_dims = meta.opt_u32("rope.dimension_count", head_dim);
    // Presence check: fail pulls early on files missing the FFN size.
    let _ffn = meta.req_u32("feed_forward_length")?;
    let eps = meta.opt_f32("attention.layer_norm_rms_epsilon", 1e-5) as f64;
    let ctx = meta.opt_u32("context_length", 4096);

    let d_conv = meta.opt_u32("ssm.conv_kernel", 4);
    let d_state = meta.opt_u32("ssm.state_size", 128);
    let n_group = meta.opt_u32("ssm.group_count", d_state);
    let dt_rank = meta.opt_u32("ssm.time_step_rank", 16);
    let d_inner = meta.opt_u32("ssm.inner_size", hidden);
    let head_k_dim = d_state;
    let head_v_dim = if dt_rank > 0 { d_inner / dt_rank } else { d_inner };
    let interval = meta.opt_u32("full_attention_interval", 4).max(1);

    // Which layers are recurrent: an explicit bool array wins, else every
    // layer whose (index + 1) isn't a multiple of the full-attn interval.
    let recurrent: Vec<bool> = meta
        .opt_bool_array("attention.recurrent_layers", n_layer)
        .unwrap_or_else(|| (0..n_layer).map(|i| (i + 1) % interval != 0).collect());

    // The MRoPE sections degenerate to a single contiguous rope for text
    // models (all section positions equal), so plain half-split rope over
    // `rope_dims` with the family's base is exact.
    let freq_base = meta.opt_f32("rope.freq_base", 1e7) as f64;
    let inv_freq = scaled_inv_freq(freq_base, rope_dims, parse_qwen35_scaling(&meta));
    let rope = std::sync::Arc::new(RopeTables::new(
        &inv_freq,
        ctx,
        DType::F32,
        device,
    )?);

    // MTP/NextN blocks are speculative-decode only — ignored if present.
    if gg.tensor("nextn.eh_proj.weight").is_ok() {
        tracing::warn!("qwen35: MTP/NextN tensors present — speculative-decode layers ignored");
    }

    let norm = gg.rms_norm("output_norm.weight", eps)?;

    let mut layers = Vec::with_capacity(n_layer);
    for il in 0..n_layer {
        let prefix = format!("blk.{il}");
        if gg.tensor(&format!("{prefix}.ffn_gate_inp.weight")).is_ok() {
            return Err(Error::Model(format!(
                "qwen35: layer {il} has MoE FFN tensors — the qwen35moe variant isn't \
                 supported yet (no released GGUF to validate against)"
            )));
        }
        let kind = if recurrent.get(il).copied().unwrap_or(true) {
            LayerKind::Recurrent(GdnAttn::new(
                &mut gg,
                &prefix,
                d_conv,
                n_group,
                dt_rank,
                head_k_dim,
                head_v_dim,
                d_inner,
                eps,
                kv_repeat_tiled,
                device,
            )?)
        } else {
            LayerKind::Full(FullAttention::new(
                &mut gg,
                &prefix,
                n_head,
                n_kv_head,
                head_dim,
                rope_dims,
                eps,
                rope.clone(),
            )?)
        };
        layers.push(Layer {
            attn_norm: gg.rms_norm(&format!("{prefix}.attn_norm.weight"), eps)?,
            post_norm: gg.rms_norm(&format!("{prefix}.post_attention_norm.weight"), eps)?,
            kind,
            ffn: Mlp {
                gate: gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?,
                up: gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?,
                down: gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?,
            },
        });
    }

    let dtype = DType::F32;
    let tok = std::sync::Arc::new(gg.tensor("token_embd.weight")?);
    Ok(Box::new(Qwen35Model {
        arch: loaded.arch.clone(),
        embeddings: tok.clone(),
        output: QMatMul::from_weights(tok)?,
        layers,
        norm,
        dtype,
    }))
}

/// qwen35 ships no rope-scaling keys today; the hook keeps the hparam future-proof.
fn parse_qwen35_scaling(meta: &Meta) -> Scaling {
    crate::model::common::parse_scaling(meta)
}

// ──────────────────────────────── tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::gguf_file::Value;

    /// 2 layers, interval 2 → layer 0 recurrent (GDN), layer 1 full attention.
    fn write_test_gguf(path: &std::path::Path) -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let vocab = 64;
        let hidden = 16;
        let heads = 2;
        let kv_heads = 1;
        let head_dim = 8;
        let rope_dims = 4;
        let layers = 2;
        let ffn = 32;
        let ctx = 128;

        let d_conv = 3;
        let d_state = 4;
        let n_group = 2;
        let dt_rank = 2;
        let inner = 8;
        let head_k_dim = d_state;
        let head_v_dim = inner / dt_rank;
        let k_total = n_group * head_k_dim;
        let conv_dim = inner + 2 * k_total;

        let meta = vec![
            ("general.architecture", Value::String("qwen35".into())),
            ("general.name", Value::String("test-qwen35".into())),
            ("qwen35.embedding_length", Value::U32(hidden as u32)),
            ("qwen35.block_count", Value::U32(layers as u32)),
            ("qwen35.context_length", Value::U32(ctx as u32)),
            ("qwen35.feed_forward_length", Value::U32(ffn as u32)),
            ("qwen35.attention.head_count", Value::U32(heads as u32)),
            ("qwen35.attention.head_count_kv", Value::U32(kv_heads as u32)),
            ("qwen35.attention.key_length", Value::U32(head_dim as u32)),
            ("qwen35.attention.value_length", Value::U32(head_dim as u32)),
            ("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("qwen35.rope.dimension_count", Value::U32(rope_dims as u32)),
            ("qwen35.rope.freq_base", Value::F32(10000.0)),
            ("qwen35.ssm.conv_kernel", Value::U32(d_conv as u32)),
            ("qwen35.ssm.state_size", Value::U32(d_state as u32)),
            ("qwen35.ssm.group_count", Value::U32(n_group as u32)),
            ("qwen35.ssm.time_step_rank", Value::U32(dt_rank as u32)),
            ("qwen35.ssm.inner_size", Value::U32(inner as u32)),
            ("qwen35.full_attention_interval", Value::U32(2)),
        ];

        // NOTE: projections are GGUF-oriented (out, in).
        let mut tensors: Vec<(String, Tensor)> = vec![
            ("token_embd.weight".into(), Tensor::randn(0f32, 1f32, (vocab, hidden), &dev)?),
            ("output_norm.weight".into(), Tensor::ones((hidden,), DType::F32, &dev)?),
        ];
        for l in 0..layers {
            let p = format!("blk.{l}");
            let common: Vec<(String, Tensor)> = vec![
                (format!("{p}.attn_norm.weight"), Tensor::ones((hidden,), DType::F32, &dev)?),
                (format!("{p}.post_attention_norm.weight"), Tensor::ones((hidden,), DType::F32, &dev)?),
                (format!("{p}.ffn_gate.weight"), Tensor::randn(0f32, 1f32, (ffn, hidden), &dev)?),
                (format!("{p}.ffn_up.weight"), Tensor::randn(0f32, 1f32, (ffn, hidden), &dev)?),
                (format!("{p}.ffn_down.weight"), Tensor::randn(0f32, 1f32, (hidden, ffn), &dev)?),
            ];
            tensors.extend(common);
            if l % 2 == 1 {
                // Full attention (interval 2 → odd layers).
                tensors.extend(vec![
                    (format!("{p}.attn_q.weight"), Tensor::randn(0f32, 1f32, (heads * 2 * head_dim, hidden), &dev)?),
                    (format!("{p}.attn_k.weight"), Tensor::randn(0f32, 1f32, (kv_heads * head_dim, hidden), &dev)?),
                    (format!("{p}.attn_v.weight"), Tensor::randn(0f32, 1f32, (kv_heads * head_dim, hidden), &dev)?),
                    (format!("{p}.attn_output.weight"), Tensor::randn(0f32, 1f32, (hidden, heads * head_dim), &dev)?),
                    (format!("{p}.attn_q_norm.weight"), Tensor::ones((head_dim,), DType::F32, &dev)?),
                    (format!("{p}.attn_k_norm.weight"), Tensor::ones((head_dim,), DType::F32, &dev)?),
                ]);
            } else {
                // Recurrent GDN.
                tensors.extend(vec![
                    (format!("{p}.attn_qkv.weight"), Tensor::randn(0f32, 1f32, (conv_dim, hidden), &dev)?),
                    (format!("{p}.attn_gate.weight"), Tensor::randn(0f32, 1f32, (inner, hidden), &dev)?),
                    // GGUF ne = [d_conv, conv_dim] → Rust (conv_dim, d_conv).
                    (format!("{p}.ssm_conv1d.weight"), Tensor::randn(0f32, 1f32, (conv_dim, d_conv), &dev)?),
                    (format!("{p}.ssm_dt.bias"), Tensor::randn(0f32, 1f32, (dt_rank,), &dev)?),
                    (format!("{p}.ssm_a"), Tensor::randn(0f32, 1f32, (dt_rank,), &dev)?),
                    (format!("{p}.ssm_beta.weight"), Tensor::randn(0f32, 1f32, (dt_rank, hidden), &dev)?),
                    (format!("{p}.ssm_alpha.weight"), Tensor::randn(0f32, 1f32, (dt_rank, hidden), &dev)?),
                    (format!("{p}.ssm_norm.weight"), Tensor::ones((head_v_dim,), DType::F32, &dev)?),
                    (format!("{p}.ssm_out.weight"), Tensor::randn(0f32, 1f32, (hidden, inner), &dev)?),
                ]);
            }
        }

        crate::model::common::gguf_test_file(path, &meta, &tensors)?;
        Ok(())
    }

    fn run_forward() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("qwen35-test.gguf");
        write_test_gguf(&path).unwrap();
        let mut loaded = LoadedModel::load(&path).unwrap();
        assert_eq!(loaded.arch, "qwen35");
        let mut model = build(&mut loaded, &Device::Cpu).unwrap();

        // Prefill.
        let input = Tensor::from_vec(vec![3u32, 1, 4, 1], (1, 4), &Device::Cpu).unwrap();
        let logits = model.forward(&input, 0).unwrap();
        assert_eq!(logits.dims(), &[1, 64], "last-position logits over vocab");
        let l = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let bad = l.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(bad, 0, "gguf qwen35 finite logits (bad {bad})");

        // Decode continues both the KV cache and the recurrent states.
        let next = Tensor::from_vec(vec![5u32], (1, 1), &Device::Cpu).unwrap();
        let logits2 = model.forward(&next, 4).unwrap();
        assert_eq!(logits2.dims(), &[1, 64]);

        // A longer prompt exercises the chunking path (> CHUNK tokens would
        // need a bigger ctx; the 4+1 steps above already cover chunk-tail
        // handling since 4 < CHUNK exercises the small-tail branch).
        model.clear_kv_cache();
        let input = Tensor::from_vec(vec![3u32, 1, 4, 1], (1, 4), &Device::Cpu).unwrap();
        let logits3 = model.forward(&input, 0).unwrap();
        let same = logits
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .zip(logits3.to_vec2::<f32>().unwrap().iter())
            .all(|(a, b)| a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-4));
        assert!(same, "clear_kv_cache makes a fresh prefill deterministic");
    }

    #[test]
    fn qwen35_end_to_end() {
        run_forward();
    }

    /// The chunked UT-transform kernel must match the trivially-correct
    /// stepwise delta rule on random data (10 tokens of 2 heads, chunks of 4
    /// vs chunks of 1).
    #[test]
    fn chunked_delta_rule_matches_stepwise() -> anyhow::Result<()> {
        let dev = Device::Cpu;
        let (h, dk, dv, t) = (2usize, 8usize, 6usize, 10usize);
        let mut rng = 0x1234u64;
        let mut rand = |rows: usize, cols: usize| -> candle_core::Result<Tensor> {
            // Cheap deterministic pseudo-random from affine+sin (no rand dep).
            let base = Tensor::arange(0u32, (rows * cols) as u32, &dev)?.to_dtype(DType::F32)?;
            let mixed = (base.affine(0.01, 0.0)?.sin()? * 3.0)?;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mixed = mixed.affine(1.0, (((rng >> 33) as f64) % 7.0) - 3.0)?;
            mixed.reshape((rows, cols))
        };
        let q = l2_norm(&rand(h * t, dk)?, 1e-6)?.reshape((h, t, dk))?; // (h, t, dk)
        let k = l2_norm(&rand(h * t, dk)?, 1e-6)?.reshape((h, t, dk))?;
        let q = q.affine(1.0 / (dk as f64).sqrt(), 0.0)?;
        let v = rand(h * t, dv)?.reshape((h, t, dv))?;
        // beta in (0,1), gate strictly negative (decay).
        let beta = candle_nn::ops::sigmoid(&rand(h * t, 1)?.reshape((h, t, 1))?)?.affine(0.9, 0.0)?;
        let gate = candle_nn::ops::sigmoid(&rand(h * t, 1)?.reshape((h, t, 1))?)?.affine(-0.1, 0.0)?;
        let s0 = Tensor::zeros((h, dk, dv), DType::F32, &dev)?;

        // Stepwise: the reference recurrence, transcribed from the HF kernel.
        let mut s = s0.clone();
        let mut step_out = Vec::new();
        for i in 0..t {
            let k_t = k.narrow(1, i, 1)?.transpose(1, 2)?; // (h, dk, 1)
            let v_t = v.narrow(1, i, 1)?.transpose(1, 2)?; // (h, dv, 1)
            let q_t = q.narrow(1, i, 1)?.transpose(1, 2)?; // (h, dk, 1)
            let b_t = beta.narrow(1, i, 1)?.reshape((h, 1, 1))?;
            let g_t = gate.narrow(1, i, 1)?.exp()?;
            let s_d = s.broadcast_mul(&g_t)?;
            let kv = s_d.transpose(1, 2)?.matmul(&k_t)?; // (h, dv, 1)
            let delta = (v_t.broadcast_sub(&kv)?).broadcast_mul(&b_t)?;
            let outer = k_t.matmul(&delta.transpose(1, 2)?)?;
            s = s_d.add(&outer)?;
            step_out.push(s.transpose(1, 2)?.matmul(&q_t)?.transpose(1, 2)?);
        }
        let stepwise = Tensor::cat(&step_out, 1)?;

        // Chunked: same code path as the model's prefill with chunk 4.
        let (chunked, _) = chunked_delta_rule(&q, &k, &v, &beta, &gate, &s0, 4)?;
        let (chunked_1, _) = chunked_delta_rule(&q, &k, &v, &beta, &gate, &s0, 1)?;
        let cmp = |a: &Tensor, b: &Tensor| -> bool {
            let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            av.iter().zip(bv.iter()).all(|(x, y)| (x - y).abs() < 1e-3)
        };
        assert!(cmp(&stepwise, &chunked_1), "stepwise == chunk(1)");
        assert!(cmp(&stepwise, &chunked), "stepwise == chunk(4)");
        Ok(())
    }

    /// The WY-solve guard: a chunk of IDENTICAL, β≈1 keys makes
    /// (I + M)⁻¹ reach 2^L-scale entries — the exact pathology that NaN'd
    /// Qwen3.5-4B prefills ≥ ~400 tokens. Two properties: (a) the output and
    /// final state stay FINITE no matter how ill-conditioned the chunks get
    /// (near-singular systems may legitimately amplify, but never explode);
    /// (b) when the guard fires, the chunk is redone with the token-recurrent
    /// form, which is bitwise the chunk-size-1 path.
    #[test]
    fn ill_conditioned_chunks_stay_finite() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let identical_keys = |h: usize, t: usize, dk: usize| {
            let ones_row = Tensor::arange(0u32, dk as u32, &dev)?
                .to_dtype(DType::F32)?
                .reshape((1, 1, dk))?
                .broadcast_as((h, t, dk))?
                .contiguous()?;
            l2_norm(&ones_row, 1e-6)
        };

        // (a) Near-singular but under the guard at chunk 4 — finite output.
        let (h, t, dk, dv) = (2usize, 8usize, 4usize, 4usize);
        let k = identical_keys(h, t, dk)?;
        let q = k.clone();
        let v = Tensor::arange(0u32, (h * t * dv) as u32, &dev)?
            .to_dtype(DType::F32)?
            .reshape((h, t, dv))?
            .affine(0.1, -1.0)?;
        let beta = Tensor::ones((h, t, 1), DType::F32, &dev)?.affine(0.999, 0.0)?;
        let gate = Tensor::full(-0.01f32, (h, t, 1), &dev)?;
        let s0 = Tensor::zeros((h, dk, dv), DType::F32, &dev)?;
        let (chunked, s_final) = chunked_delta_rule(&q, &k, &v, &beta, &gate, &s0, 4)?;
        let finite = |t: &Tensor| {
            t.mul(t).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().is_finite()
        };
        assert!(finite(&chunked), "guarded chunked output must be finite");
        assert!(finite(&s_final), "guarded state must stay finite");

        // (b) The fallback wiring is exact: with the guard threshold forced
        // to ~0 every chunk takes the stepwise path, which must equal the
        // chunk-size-1 run bitwise (the same step_delta kernel on the same
        // views) on ordinary data.
        let (h, t) = (2usize, 24usize);
        let rnd = |n: usize| -> candle_core::Result<Tensor> {
            Tensor::from_vec(
                (0..n).map(|i| ((i as i32 * 73 % 97) as f32 - 48.0) / 48.0).collect(),
                n,
                &dev,
            )
        };
        let k = l2_norm(&rnd(h * t * dk)?.reshape((h, t, dk))?, 1e-6)?;
        let q = l2_norm(&rnd(h * t * dk)?.reshape((h, t, dk))?, 1e-6)?;
        let v = rnd(h * t * dv)?.reshape((h, t, dv))?;
        let beta = rnd(h * t)?.reshape((h, t, 1))?.affine(0.49, 0.5)?;
        let gate = rnd(h * t)?.reshape((h, t, 1))?.affine(0.05, -0.02)?;
        let (forced, _) =
            chunked_delta_rule_guarded(&q, &k, &v, &beta, &gate, &s0, 8, 0.001)?;
        let (pure_step, _) = chunked_delta_rule(&q, &k, &v, &beta, &gate, &s0, 1)?;
        let av = forced.flatten_all()?.to_vec1::<f32>()?;
        let bv = pure_step.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            av.iter().zip(bv.iter()).all(|(x, y)| x == y),
            "forced fallback must BE the stepwise path (bitwise)"
        );
        assert!(
            av.iter().all(|x| x.is_finite()),
            "fallback output must be finite"
        );
        Ok(())
    }

    /// The k→v head duplication must place the copy axis per the weight
    /// source: GGUF (llama.cpp converter) TILES — head j pairs with
    /// k[j % n_k]; raw HF safetensors BLOCK (repeat_interleave) — heads
    /// [2i, 2i+1] pair with k[i]. Putting the axis between head and time
    /// instead interleaves it with TIME and scrambles q/k silently (the
    /// Qwen3.5-4B incident — first model with n_v != n_k). See ACRoad §6b.
    #[test]
    fn kv_head_grow_layouts() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let (n_k, rep, t_len, d) = (2usize, 2usize, 2usize, 2usize);
        let base = Tensor::arange(0u32, (n_k * t_len * d) as u32, &dev)?
            .to_dtype(DType::F32)?
            .reshape((n_k, t_len, d))?; // heads: [[[0,1],[2,3]], [[4,5],[6,7]]]
        let v = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // TILED (GGUF): [k0, k1, k0, k1] — rep axis OUTSIDE the head axis.
        let tiled = base
            .reshape((1, n_k, t_len, d))?
            .broadcast_as((rep, n_k, t_len, d))?
            .contiguous()?
            .reshape((n_k * rep, t_len, d))?;
        assert_eq!(
            v(&tiled),
            vec![0., 1., 2., 3., 4., 5., 6., 7., 0., 1., 2., 3., 4., 5., 6., 7.],
            "tiled grow repeats whole k-head groups"
        );

        // BLOCK (HF / repeat_interleave): [k0, k0, k1, k1] — rep axis INSIDE.
        let block = base
            .reshape((n_k, 1, t_len, d))?
            .broadcast_as((n_k, rep, t_len, d))?
            .contiguous()?
            .reshape((n_k * rep, t_len, d))?;
        assert_eq!(
            v(&block),
            vec![0., 1., 2., 3., 0., 1., 2., 3., 4., 5., 6., 7., 4., 5., 6., 7.],
            "block grow duplicates each k head in place"
        );

        // The BUG (rep axis between head and time) must NOT match either.
        let scrambled = base
            .reshape((n_k, t_len, 1, d))?
            .broadcast_as((n_k, t_len, rep, d))?
            .contiguous()?
            .reshape((n_k * rep, t_len, d))?;
        assert_ne!(v(&scrambled), v(&tiled));
        assert_ne!(v(&scrambled), v(&block));
        Ok(())
    }

    /// Debug dump: read the real model's small GDN tensors (env-gated so CI
    /// machines without the file skip it).
    #[test]
    fn dump_real_gdn_tensors() {
        let Ok(path) = std::env::var("QWEN35_REAL") else { return };
        let mut loaded = LoadedModel::load(std::path::Path::new(&path)).unwrap();
        let content = loaded.take_content().unwrap();
        // Tensor dtype census for one GDN layer + one full-attn layer + the
        // globals — quantized small tensors (norms, SSM scalars) are a
        // classic silent-corruption source in imatrix quants.
        let census: Vec<(String, String)> = content
            .tensor_infos
            .iter()
            .map(|(k, ti)| (k.clone(), format!("{:?} {:?}", ti.ggml_dtype, ti.shape.dims())))
            .collect();
        for (k, v) in census {
            if !k.starts_with("blk.") || k.starts_with("blk.0.") || k.starts_with("blk.3.") {
                println!("{k}: {v}");
            }
        }
        let mut gg = Gguf::new(content, &mut loaded.file, Device::Cpu);
        for name in ["blk.0.ssm_a", "blk.0.ssm_dt.bias"] {
            let t = gg.tensor(name).unwrap().dequantize(&Device::Cpu).unwrap();
            let v = t.to_vec1::<f32>().unwrap();
            println!("{name}: {v:?}");
        }
        let conv = gg.tensor("blk.0.ssm_conv1d.weight").unwrap().dequantize(&Device::Cpu).unwrap();
        println!("conv1d dims: {:?}", conv.dims());
    }
}

#[cfg(test)]
mod cuda_prim_tests {
    use candle_core::{DType, Tensor};

    /// Cumsum/mask primitives of the chunked delta rule, CUDA vs CPU, at the
    /// sizes the prefill actually hits (L = 6 worked, L = 64 NaN'd on CUDA).
    #[test]
    fn cuda_cumsum_and_masks_match_cpu() {
        let Ok(cuda) = candle_core::Device::new_cuda(0) else {
            eprintln!("no cuda — skipping");
            return;
        };
        for l in [6usize, 16, 63, 64, 65] {
            // 1. The causal mask builder: eye(L).cumsum(0).
            let cpu_causal = Tensor::eye(l, DType::F32, &candle_core::Device::Cpu)
                .unwrap()
                .cumsum(0)
                .unwrap();
            let cu_causal = Tensor::eye(l, DType::F32, &cuda).unwrap().cumsum(0).unwrap();
            let cu_back = cu_causal.to_device(&candle_core::Device::Cpu).unwrap();
            let diff = cu_back
                .sub(&cpu_causal)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(diff, 0.0, "causal mask L={l}: cuda vs cpu max diff {diff}");

            // 2. The g_cum pattern: (n_v, L, 1).cumsum(1) — non-contiguous
            // transpose through broadcast_matmul.
            let n_v = 32usize;
            let data: Vec<f32> = (0..n_v * l).map(|i| (i % 13) as f32 * 0.25).collect();
            let cpu_g = Tensor::from_slice(&data, (n_v, l, 1), &candle_core::Device::Cpu)
                .unwrap()
                .cumsum(1)
                .unwrap();
            let cu_g = Tensor::from_slice(&data, (n_v, l, 1), &cuda)
                .unwrap()
                .cumsum(1)
                .unwrap()
                .to_device(&candle_core::Device::Cpu)
                .unwrap();
            let diff = cu_g
                .sub(&cpu_g)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(diff, 0.0, "batched cumsum L={l}: cuda vs cpu max diff {diff}");
        }
    }

    /// The (I+M)⁻¹ nilpotent doubling on CUDA: exact integer-valued series
    /// make any divergence obvious.
    #[test]
    fn cuda_nilpotent_doubling_matches_cpu() {
        let Ok(cuda) = candle_core::Device::new_cuda(0) else {
            eprintln!("no cuda — skipping");
            return;
        };
        for l in [6usize, 16, 64] {
            // M = strict_lower of an integer matrix — (I+M)⁻¹ has integer
            // entries (alternating signed powers), exact in f32 at these mags.
            let cpu = candle_core::Device::Cpu;
            let data: Vec<f32> = (0..l * l)
                .map(|i| if i % l != 0 && i % l < i / l { ((i % 5) as f32) - 2.0 } else { 0.0 })
                .collect();
            let m_cpu = Tensor::from_slice(&data, (l, l), &cpu).unwrap();
            let m_cu = Tensor::from_slice(&data, (l, l), &cuda).unwrap();

            let inv = |m: &Tensor, dev: &candle_core::Device| -> candle_core::Result<Tensor> {
                let neg_m = m.affine(-1.0, 0.0)?;
                let eye = Tensor::eye(l, DType::F32, dev)?
                    .broadcast_as((1, l, l))?
                    .contiguous()?;
                let mut t_series = eye;
                let mut p_k = neg_m.unsqueeze(0)?.contiguous()?;
                let mut m_len = 1usize;
                while m_len < l {
                    t_series = t_series.broadcast_matmul(&p_k)?.broadcast_add(&t_series)?;
                    p_k = p_k.broadcast_matmul(&p_k)?;
                    m_len *= 2;
                }
                t_series.squeeze(0)
            };
            let a = inv(&m_cpu, &cpu).unwrap();
            let b = inv(&m_cu, &cuda)
                .unwrap()
                .to_device(&cpu)
                .unwrap();
            let diff = b.sub(&a).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert_eq!(diff, 0.0, "nilpotent doubling L={l}: max diff {diff}");
        }
    }
}
