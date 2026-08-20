//! Logit sampling strategies.
//!
//! Wraps [`candle_transformers::generation::LogitsProcessor`] — the canonical
//! candle sampler — behind AmberCore's own [`SampleParams`]. This gives us
//! greedy, temperature, top-k, and top-p sampling with correct numerics
//! (softmax + multinomial) without hand-rolling them.
//!
//! The processor takes a logits [`Tensor`] and returns the sampled token id.
//! The pipeline owns one processor per generation session (the RNG state
//! carries across tokens for reproducibility given a seed).

use crate::error::Result;
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};

/// Sampling parameters. Translated to a candle [`Sampling`] strategy on build.
#[derive(Debug, Clone)]
pub struct SampleParams {
    /// Temperature: `< 1.0` sharpens, `> 1.0` flattens. `<= 0` means greedy.
    pub temperature: f32,
    /// Top-k: only consider the k highest-probability tokens (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus): smallest set whose cumulative prob ≥ p (>= 1.0 = disabled).
    pub top_p: f32,
    /// RNG seed for reproducible stochastic sampling.
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 0,
            top_p: 0.95,
            seed: 299792458,
        }
    }
}

impl SampleParams {
    /// Build a candle [`LogitsProcessor`] from these params.
    ///
    /// Strategy selection:
    /// - `temperature <= 0` → greedy argmax
    /// - `top_k > 0` and `top_p < 1.0` → top-k then top-p
    /// - `top_k > 0` → top-k
    /// - `top_p < 1.0` → top-p (nucleus)
    /// - else → plain temperature sampling
    pub fn build_processor(&self) -> LogitsProcessor {
        let temp = if self.temperature <= 0.0 {
            None
        } else {
            Some(self.temperature as f64)
        };
        let sampling = if temp.is_none() {
            Sampling::ArgMax
        } else {
            let t = temp.unwrap();
            match (self.top_k, self.top_p < 1.0) {
                (0, false) => Sampling::All { temperature: t },
                (k, false) => Sampling::TopK { k, temperature: t },
                (0, true) => Sampling::TopP {
                    p: self.top_p as f64,
                    temperature: t,
                },
                (k, true) => Sampling::TopKThenTopP {
                    k,
                    p: self.top_p as f64,
                    temperature: t,
                },
            }
        };
        LogitsProcessor::from_sampling(self.seed, sampling)
    }
}

/// A token sampler. Owns the candle [`LogitsProcessor`] for one session so the
/// RNG state advances correctly across tokens.
pub struct Sampler {
    processor: LogitsProcessor,
}

impl Sampler {
    pub fn new(params: &SampleParams) -> Self {
        Self {
            processor: params.build_processor(),
        }
    }

    /// Sample one token id from the last-position logits `[batch, vocab]`.
    /// The batch dim must be 1; we squeeze it off before sampling.
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits
            .squeeze(0)
            .map_err(|e| crate::error::Error::Model(format!("squeeze logits: {e}")))?;
        let id = self
            .processor
            .sample(&logits)
            .map_err(|e| crate::error::Error::Model(format!("sample: {e}")))?;
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn greedy_sampler_picks_argmax() {
        // A 1-batch, 5-vocab logits tensor with index 3 as the max.
        let logits = Tensor::from_iter([0.1f32, 0.2, -1.0, 5.0, 0.0], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let params = SampleParams {
            temperature: 0.0, // greedy
            ..Default::default()
        };
        let mut sampler = Sampler::new(&params);
        assert_eq!(sampler.sample(&logits).unwrap(), 3);
    }

    #[test]
    fn greedy_is_deterministic_across_calls() {
        let logits = Tensor::from_iter([0.1f32, 0.9, 0.2], &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let params = SampleParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(&params);
        let a = sampler.sample(&logits).unwrap();
        let b = sampler.sample(&logits).unwrap();
        assert_eq!(a, b, "greedy must be deterministic");
    }

    #[test]
    fn stochastic_sampler_seeded_is_reproducible() {
        // Two samplers with the same seed + temperature must produce the same
        // sequence (verifies RNG state is seeded, not wall-clock-derived).
        let make_logits = || {
            Tensor::from_iter(
                [0.5f32, 0.3, 0.8, 0.1, 0.6, 0.4, 0.2, 0.7, 0.55, 0.45],
                &Device::Cpu,
            )
            .unwrap()
            .unsqueeze(0)
            .unwrap()
        };
        let params = SampleParams {
            temperature: 0.7,
            top_p: 0.9,
            seed: 12345,
            ..Default::default()
        };
        let mut s1 = Sampler::new(&params);
        let mut s2 = Sampler::new(&params);
        for _ in 0..10 {
            let l = make_logits();
            assert_eq!(s1.sample(&l).unwrap(), s2.sample(&make_logits()).unwrap());
        }
    }
}
