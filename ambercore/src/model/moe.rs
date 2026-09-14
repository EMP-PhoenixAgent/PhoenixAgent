//! Routed mixture-of-experts FFN for the GGUF adapters.
//!
//! candle's [`FusedMoeGGUF`](candle_transformers::fused_moe) hard-codes softmax
//! routing, but three of the families AmberCore loads route differently:
//!
//! - **deepseek2 / deepseek32 (DeepSeek V2.5+/V3/R1 and Kimi K2)**: sigmoid
//!   gating, optional `ffn_exp_probs_b` gate bias, top-k weight renorm, and a
//!   multiplicative `routed_scaling_factor` (`expert_weights_scale`).
//! - **minimax-m2**: sigmoid gating with a mandatory gate bias.
//! - **granitemoe / gemma4-MoE**: softmax with renorm.
//!
//! [`RoutedMoe`] is that union over candle's public `moe_gemm_gguf` grouped-GEMM
//! primitive (the same engine [`crate::model::qwen3_moe`] rides), so expert
//! weights stay quantized on device — no per-expert dequantization. The shared
//! expert (`ffn_*_shexp`) is intentionally **not** handled here: its placement
//! (added before/after residual, parallel MLP in gemma4) differs per family, so
//! each adapter keeps it inline.

use candle_core::quantized::{QStorage, QTensor};
use candle_core::{DType, Result, Tensor};
use candle_nn::{moe, Activation, Linear, Module};
use candle_transformers::models::with_tracing::QMatMul;
use std::borrow::Cow;
use std::sync::Arc;

/// How router logits turn into expert probabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gating {
    /// `softmax(logits + bias)` — qwen, granite-moe, gemma4-MoE.
    Softmax,
    /// `sigmoid(logits + bias)` — DeepSeek V3/R1/V3.2, Kimi K2, MiniMax M2.
    Sigmoid,
}

/// The expert weight stacks, held in whichever form the device can execute.
///
/// - **CUDA**: the 3-D grouped tensors stay whole and run through candle's
///   grouped `moe_gemm_gguf` (the same kernel [`crate::model::qwen3_moe`]
///   rides).
/// - **CPU**: candle's grouped GEMM is CUDA-only, so each expert's slice of
///   the 3-D stack is extracted **still quantized** (a contiguous byte range
///   rewrapped via `QStorage::from_data` — no dequantization, no extra
///   memory) into its own `QMatMul`, and the forward loops over experts.
///   Decode steps touch only the top-k experts of one token; prefill loops
///   all experts but gathers just their routed tokens.
enum ExpertStack {
    Grouped {
        gate: Arc<QTensor>,
        up: Arc<QTensor>,
        down: Arc<QTensor>,
    },
    PerExpert {
        gate: Vec<QMatMul>,
        up: Vec<QMatMul>,
        down: Vec<QMatMul>,
    },
}

impl ExpertStack {
    /// Split a 3-D `(n_experts, n, k)` quantized stack into per-expert
    /// `QMatMul`s on the CPU. Expert `e` occupies a contiguous byte range
    /// (the expert dim is outermost), so this is a copy of quantized bytes —
    /// the same footprint as the stack, which is then dropped.
    fn extract_per_expert(stack: &QTensor, num_experts: usize, device: &candle_core::Device) -> Result<Vec<QMatMul>> {
        let dtype = stack.dtype();
        let total = stack.storage_size_in_bytes();
        let per_expert = total / num_experts;
        if per_expert * num_experts != total {
            candle_core::bail!("expert stack byte size {total} not divisible by {num_experts} experts");
        }
        let data = stack.data()?;
        let (n, k) = (stack.shape().dims()[1], stack.shape().dims()[2]);
        let mut out = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let bytes = &data[e * per_expert..(e + 1) * per_expert];
            let storage = QStorage::from_data(Cow::Owned(bytes.to_vec()), device, dtype)?;
            let qt = QTensor::new(storage, (n, k))?;
            out.push(QMatMul::from_weights(Arc::new(qt))?);
        }
        Ok(out)
    }
}

/// A routed-MoE layer loaded from GGUF 3-D expert tensors.
pub struct RoutedMoe {
    gate: Linear,
    experts: ExpertStack,
    act: Activation,
    gating: Gating,
    gate_bias: Option<Tensor>,
    /// Renormalize the selected top-k weights to sum to 1 (DeepSeek's
    /// `norm_topk_prob`, qwen's shared-expert-less renorm).
    norm_topk: bool,
    /// Multiplicative routed-weights scale (DeepSeek `routed_scaling_factor`).
    weights_scale: f64,
    num_experts_per_tok: usize,
    num_experts: usize,
    dtype: DType,
}

impl RoutedMoe {
    /// Load from GGUF tensor names under `prefix` (`blk.N`), quantized experts
    /// included. `gate`/`gate_bias` are dequantized to F32 (router matrices are
    /// tiny). The bias tensor is optional and read as `ffn_exp_probs_b.bias`.
    pub fn load<R: std::io::Seek + std::io::Read>(
        gg: &mut candle_transformers::models::quantized_qwen3::Gguf<R>,
        prefix: &str,
        hidden_size: usize,
        num_experts: usize,
        moe_intermediate_size: usize,
        num_experts_per_tok: usize,
        act: Activation,
        gating: Gating,
        norm_topk: bool,
        weights_scale: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        // Router in F32 like candle's FusedMoeGGUF: the dequantized GGUF
        // tensor is (num_experts, hidden) — already candle's Linear (out, in).
        let gate_ws = gg
            .tensor(&format!("{prefix}.ffn_gate_inp.weight"))?
            .dequantize(device)?
            .to_dtype(DType::F32)?;
        if gate_ws.dims() != [num_experts, hidden_size] {
            candle_core::bail!(
                "{prefix}.ffn_gate_inp is {:?}, expected [{num_experts}, {hidden_size}]",
                gate_ws.dims()
            );
        }
        let gate = Linear::new(gate_ws, None);

        let gate_bias = match gg.tensor(&format!("{prefix}.ffn_exp_probs_b.bias")) {
            Ok(t) => Some(t.dequantize(device)?.to_dtype(DType::F32)?),
            Err(_) => None,
        };

        let gate_experts = gg.tensor(&format!("{prefix}.ffn_gate_exps.weight"))?;
        let up_experts = gg.tensor(&format!("{prefix}.ffn_up_exps.weight"))?;
        let down_experts = gg.tensor(&format!("{prefix}.ffn_down_exps.weight"))?;
        // Expert stacks are (n_experts, intermediate, hidden) for gate/up and
        // (n_experts, hidden, intermediate) for down — validate against the
        // metadata so a mismatched GGUF fails here with a readable message
        // instead of deep inside the GEMM.
        for (name, want, got) in [
            ("ffn_gate_exps", [num_experts, moe_intermediate_size, hidden_size], gate_experts.shape().dims()),
            ("ffn_up_exps", [num_experts, moe_intermediate_size, hidden_size], up_experts.shape().dims()),
            ("ffn_down_exps", [num_experts, hidden_size, moe_intermediate_size], down_experts.shape().dims()),
        ] {
            if got != want {
                candle_core::bail!("{prefix}.{name} is {got:?}, expected {want:?}");
            }
        }

        // CUDA keeps the grouped stacks (grouped GEMM kernel); CPU splits
        // them into per-expert QMatMuls (see `ExpertStack`).
        let experts = match device {
            candle_core::Device::Cuda(_) => ExpertStack::Grouped {
                gate: Arc::new(gate_experts),
                up: Arc::new(up_experts),
                down: Arc::new(down_experts),
            },
            _ => ExpertStack::PerExpert {
                gate: ExpertStack::extract_per_expert(&gate_experts, num_experts, device)?,
                up: ExpertStack::extract_per_expert(&up_experts, num_experts, device)?,
                down: ExpertStack::extract_per_expert(&down_experts, num_experts, device)?,
            },
        };

        Ok(Self {
            gate,
            experts,
            act,
            gating,
            gate_bias,
            norm_topk,
            weights_scale,
            num_experts_per_tok,
            num_experts,
            dtype: DType::F32,
        })
    }

    /// Router probabilities for `xs` `(tokens, hidden)` → `(tokens, n_experts)`.
    fn route(&self, xs: &Tensor) -> Result<Tensor> {
        let logits = self.gate.forward(xs)?;
        let logits = match &self.gate_bias {
            Some(b) => logits.broadcast_add(b)?,
            None => logits,
        };
        match self.gating {
            Gating::Softmax => candle_nn::ops::softmax_last_dim(&logits),
            Gating::Sigmoid => candle_nn::ops::sigmoid(&logits),
        }
    }

    pub fn forward(&self, xs: &Tensor, is_prefill: bool) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let (num_tokens, _) = xs.dims2()?;
        let original_dtype = xs.dtype();
        let xs = if xs.dtype() != DType::F32 { xs.to_dtype(DType::F32)? } else { xs.to_owned() };

        let probs = self.route(&xs)?;

        let topk_ids = probs
            .arg_sort_last_dim(false)?
            .narrow(candle_core::D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let mut topk_weights = probs.gather(&topk_ids, candle_core::D::Minus1)?;
        if self.norm_topk {
            topk_weights = topk_weights.broadcast_div(&topk_weights.sum_keepdim(candle_core::D::Minus1)?)?;
        }
        if self.weights_scale != 1.0 {
            topk_weights = (topk_weights * self.weights_scale)?;
        }

        let ys = match &self.experts {
            ExpertStack::Grouped { gate, up, down } => {
                let (expert_ids, sorted_token_ids) = topk_ids.flatten_all()?.sort_last_dim(true)?;
                let gate = moe::moe_gemm_gguf(
                    &xs,
                    gate,
                    &None,
                    &sorted_token_ids,
                    &expert_ids,
                    self.num_experts_per_tok,
                    is_prefill,
                    self.dtype,
                )?;
                let up = moe::moe_gemm_gguf(
                    &xs,
                    up,
                    &None,
                    &sorted_token_ids,
                    &expert_ids,
                    self.num_experts_per_tok,
                    is_prefill,
                    self.dtype,
                )?;
                let down_inputs = (up * gate.apply(&self.act)?)?;
                moe::moe_gemm_gguf(
                    &down_inputs,
                    down,
                    &Some(topk_weights),
                    &sorted_token_ids,
                    &expert_ids,
                    self.num_experts_per_tok,
                    is_prefill,
                    self.dtype,
                )?
                .reshape((num_tokens, (), hidden_dim))?
                .sum(candle_core::D::Minus2)?
            }
            ExpertStack::PerExpert { gate, up, down } => {
                // CPU path: loop experts, gather that expert's (token, slot)
                // positions, run three small quantized matmuls, scatter-add.
                let ids = topk_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
                let weights = topk_weights.flatten_all()?.to_vec1::<f32>()?;
                let mut out = Tensor::zeros((num_tokens, hidden_dim), DType::F32, xs.device())?;
                for e in 0..self.num_experts {
                    // (token, slot) pairs routed to expert e.
                    let routed: Vec<(usize, usize)> = ids
                        .iter()
                        .enumerate()
                        .flat_map(|(t, row)| {
                            row.iter().enumerate().filter_map(move |(j, &id)| (id as usize == e).then_some((t, j)))
                        })
                        .collect();
                    if routed.is_empty() {
                        continue;
                    }
                    let token_idx = Tensor::from_vec(
                        routed.iter().map(|(t, _)| *t as u32).collect::<Vec<_>>(),
                        (routed.len(),),
                        xs.device(),
                    )?;
                    let x_e = xs.index_select(&token_idx.to_dtype(DType::U32)?, 0)?;
                    let g = gate[e].forward(&x_e)?.apply(&self.act)?;
                    let u = up[e].forward(&x_e)?;
                    let h = (g * u)?;
                    let y = down[e].forward(&h)?; // (routed, hidden)

                    let w: Vec<f32> = routed
                        .iter()
                        .map(|(t, j)| weights[t * self.num_experts_per_tok + j])
                        .collect();
                    let w = Tensor::from_vec(w, (routed.len(), 1), xs.device())?;
                    let w = w.broadcast_as(y.dims())?;
                    let y = (y * w)?; // (routed, hidden)
                    let dst = Tensor::from_vec(
                        routed.iter().map(|(t, _)| *t as u32).collect::<Vec<_>>(),
                        (routed.len(),),
                        xs.device(),
                    )?;
                    // out[dst[i]] += y[i] — duplicate destinations accumulate.
                    out = out.index_add(&dst, &y, 0)?;
                }
                out
            }
        };

        let ys = if ys.dtype() != original_dtype { ys.to_dtype(original_dtype)? } else { ys };
        ys.reshape((batch, seq_len, hidden_dim))
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    /// RoutedMoe is a struct over already-loaded pieces; exercise the routing
    /// math (softmax vs sigmoid, bias, renorm, scale) with hand-built tensors.
    #[test]
    fn routing_gating_math() {
        let dev = Device::Cpu;
        // 1 token, hidden 4, 3 experts. Gate rows per expert (out, in).
        let gate_w = Tensor::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
            (3, 4),
            &dev,
        )
        .unwrap();
        let xs = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (1, 4), &dev).unwrap();
        let logits = gate_w.matmul(&xs.t().unwrap()).unwrap(); // (3 experts, 1)

        // softmax over experts
        let sm = candle_nn::ops::softmax_last_dim(&logits.reshape((1, 3)).unwrap()).unwrap();
        let sm = sm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sum: f32 = sm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax normalizes");

        // sigmoid stays per-expert independent
        let sg = candle_nn::ops::sigmoid(&logits.reshape((1, 3)).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(sg[0] > 0.7 && sg[0] < 0.8, "sigmoid(1) ≈ 0.731, got {}", sg[0]);
        assert!(sg[1] > 0.87 && sg[1] < 0.89, "sigmoid(2) ≈ 0.881, got {}", sg[1]);
    }
}
