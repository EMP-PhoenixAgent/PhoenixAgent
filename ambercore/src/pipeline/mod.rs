//! Generation pipeline.
//!
//! Drives a loaded model token-by-token. The pipeline is backend- and
//! architecture-agnostic: it speaks only to the [`DynModel`] trait, the
//! [`Backend`](crate::backend::Backend) trait, and the
//! [`TokenizerWrapper`](crate::tokenizer::TokenizerWrapper).
//!
//! Two entry points:
//! - [`Pipeline::decode_one`] (M0) — a single-token forward pass; the smallest
//!   end-to-end proof.
//! - [`Pipeline::generate`] (M1) — the full streaming loop: prefill the prompt,
//!   then repeatedly decode one token (reusing the model's internal KV cache
//!   via `index_pos`), sample, stream the stable text prefix, stop on EOS or
//!   max tokens.
//!
//! [`DynModel`]: crate::model::DynModel

pub mod kv_cache;
pub mod sampler;

pub use sampler::SampleParams;

use crate::error::{Error, Result};
use crate::model::DynModel;
use crate::tokenizer::{StreamingDecoder, TokenizerWrapper};
use candle_core::Tensor;
use sampler::Sampler;

/// A single generated token: its id and the decoded text delta.
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u32,
    pub text: String,
}

/// Statistics from a generation run.
#[derive(Debug, Clone, Default)]
pub struct GenStats {
    /// Number of tokens generated (excluding the prompt).
    pub output_tokens: usize,
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Wall-clock time spent in the prefill step (processing the prompt).
    pub prefill_secs: f64,
    /// Wall-clock time spent in the decode loop.
    pub decode_secs: f64,
    /// **Time to first token** — wall-clock from the start of prefill to the
    /// moment the first token was sampled (milliseconds). `None` if no tokens
    /// were generated. (Prometheus `TTFT` event.)
    pub ttft_ms: Option<f64>,
    /// Per-step inter-token deltas during the decode loop (milliseconds). The
    /// average of these is the **time between tokens (TBT)**. Empty if only the
    /// prefill token was produced. (Drives Prometheus `GEN_STEP` events.)
    pub token_intervals_ms: Vec<f64>,
}

impl GenStats {
    /// Tokens per second during the decode phase.
    pub fn tokens_per_sec(&self) -> f64 {
        if self.decode_secs > 0.0 {
            self.output_tokens as f64 / self.decode_secs
        } else {
            0.0
        }
    }

    /// Average **time between tokens** (milliseconds) over the decode loop.
    /// `None` if fewer than two tokens were produced (no inter-token interval).
    pub fn tbt_avg_ms(&self) -> Option<f64> {
        if self.token_intervals_ms.is_empty() {
            None
        } else {
            Some(self.token_intervals_ms.iter().sum::<f64>() / self.token_intervals_ms.len() as f64)
        }
    }
}

/// The generation pipeline. Borrows the model + tokenizer for a generation
/// session and the [`Device`] the model lives on (so input tensors are placed
/// on the correct compute target).
pub struct Pipeline<'a> {
    pub model: &'a mut dyn DynModel,
    pub tokenizer: &'a TokenizerWrapper,
    pub device: &'a candle_core::Device,
    /// The model's trained context length (max prompt tokens). AmberCore never
    /// hard-blocks on this — it's local, so we let the prompt exceed it if the
    /// user wants — but we `tracing::warn` so they know quality may degrade.
    pub context_length: usize,
}

/// Stop conditions for a generation run.
///
/// AmberCore is fully local — there's no per-token billing or remote quota, so
/// the defaults are **unlimited** (`max_tokens: None`). A cap can still be set
/// explicitly (e.g. a CLI `--max-tokens N`), but nothing imposes one by default.
#[derive(Debug, Clone)]
pub struct StopCondition {
    /// Optional hard cap on generated tokens. `None` = unlimited (stop only on
    /// EOS / `<|im_end|>`). The model's natural end-of-turn is the real limit.
    pub max_tokens: Option<usize>,
    /// Token ids that end generation (EOS + assistant-turn-end markers).
    pub stop_tokens: Vec<u32>,
}

impl Default for StopCondition {
    fn default() -> Self {
        Self {
            max_tokens: None,
            stop_tokens: Vec::new(),
        }
    }
}

impl<'a> Pipeline<'a> {
    /// M0: decode exactly one token from a prompt. The smallest end-to-end win.
    ///
    /// Encodes the prompt, runs a single forward pass (prefill) at position 0,
    /// argmaxes the last position's logits, and decodes the resulting token id
    /// to text. Returns the generated [`Token`] plus the prompt length.
    pub fn decode_one(&mut self, prompt: &str) -> Result<(Token, usize)> {
        let device = self.device;
        // Start from a clean KV cache — matters when the model is reused across
        // sessions (qwen3 does not self-reset). See DynModel::clear_kv_cache.
        self.model.clear_kv_cache();
        let enc = self.tokenizer.encode(prompt)?;
        if enc.ids.is_empty() {
            return Err(Error::InvalidInput("empty prompt encoding".into()));
        }
        let prompt_len = enc.ids.len();

        let input = Tensor::new(enc.ids.as_slice(), &device)?
            .unsqueeze(0)
            .map_err(|e| Error::Model(format!("unsqueeze: {e}")))?;
        let logits = self.model.forward(&input, 0)?;
        let id = Sampler::new(&SampleParams::default()).sample(&logits)?;
        let text = self.tokenizer.decode(&[id])?;
        Ok((Token { id, text }, prompt_len))
    }

    /// M1: stream a generation, calling `on_token` for each newly-stable text
    /// delta as it lands.
    ///
    /// Flow:
    /// 1. **Prefill** — encode the prompt, run one forward pass over all tokens
    ///    at `index_pos = 0`. Sample the first generated token.
    /// 2. **Decode loop** — feed the just-sampled token as a single-token input
    ///    at `index_pos = N`, where N advances by one each step. The model's
    ///    internal KV cache makes each step O(1) in sequence length.
    /// 3. **Stream** — feed each token to the [`StreamingDecoder`] and emit the
    ///    stable prefix. Stop on a stop token or `max_tokens`.
    ///
    /// Returns the final stats + the full decoded text.
    pub fn generate<F>(
        &mut self,
        prompt: &str,
        params: &SampleParams,
        stop: &StopCondition,
        mut on_token: F,
    ) -> Result<(GenStats, String)>
    where
        F: FnMut(&str),
    {
        let device = self.device;
        let mut sampler = Sampler::new(params);
        let mut decoder = StreamingDecoder::new();

        // Reset the KV cache so a reused model starts this session clean — qwen3
        // would otherwise leak the previous sequence's K/V (garbage + memory growth).
        // See DynModel::clear_kv_cache.
        self.model.clear_kv_cache();

        // 1. Prefill.
        let enc = self.tokenizer.encode(prompt)?;
        if enc.ids.is_empty() {
            return Err(Error::InvalidInput("empty prompt encoding".into()));
        }
        let prompt_tokens = enc.ids.len();

        // Context-length awareness: AmberCore never hard-blocks (it's local —
        // no reason to refuse a long prompt), but warn when the prompt exceeds
        // the model's trained context so the user knows quality may degrade.
        if prompt_tokens > self.context_length {
            tracing::warn!(
                prompt_tokens,
                context_length = self.context_length,
                "prompt exceeds the model's trained context length; generation \
                 will proceed but quality may degrade past the training horizon"
            );
        }

        let prefill_start = std::time::Instant::now();
        let input = Tensor::new(enc.ids.as_slice(), &device)?
            .unsqueeze(0)
            .map_err(|e| Error::Model(format!("prefill input: {e}")))?;
        let logits = self.model.forward(&input, 0)?;
        let mut next_token = sampler.sample(&logits)?;
        let prefill_secs = prefill_start.elapsed().as_secs_f64();
        // TTFT: prefill start → first sampled token (includes the prefill
        // forward pass + the first sample). This is the latency a user feels
        // before the first character appears.
        let ttft_ms = Some(prefill_start.elapsed().as_secs_f64() * 1000.0);

        // Emit the first token's stable text (if any).
        if let Some(delta) = decoder.next_token(self.tokenizer, next_token)? {
            on_token(&delta);
        }

        let mut output_tokens = 1usize;
        let mut full_text = String::new();
        if let Some(d) = decoder.decode_rest(self.tokenizer)? {
            // (don't flush rest yet — generation continues)
            let _ = d;
        }

        // Check EOS on the first token.
        let mut hit_stop = stop.stop_tokens.contains(&next_token);

        // 2. Decode loop. Runs until EOS or the optional max-tokens cap.
        let decode_start = std::time::Instant::now();
        let mut step_start = std::time::Instant::now();
        let mut token_intervals_ms: Vec<f64> = Vec::new();
        let cap_hit = |n: usize| match stop.max_tokens {
            Some(cap) => n >= cap,
            None => false, // unlimited
        };
        while !cap_hit(output_tokens) && !hit_stop {
            let index_pos = prompt_tokens + output_tokens - 1;
            let input = Tensor::new(&[next_token], &device)?
                .unsqueeze(0)
                .map_err(|e| Error::Model(format!("decode input: {e}")))?;
            let logits = self.model.forward(&input, index_pos)?;
            next_token = sampler.sample(&logits)?;

            // Per-step inter-token delta (TBT material). Captured for every
            // decode iteration after the first prefill token.
            token_intervals_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);
            step_start = std::time::Instant::now();

            output_tokens += 1;
            if stop.stop_tokens.contains(&next_token) {
                hit_stop = true;
            }

            // Stream the stable prefix.
            if let Some(delta) = decoder.next_token(self.tokenizer, next_token)? {
                full_text.push_str(&delta);
                on_token(&delta);
            }
        }
        let decode_secs = decode_start.elapsed().as_secs_f64();

        // Flush any remaining text.
        if let Some(rest) = decoder.decode_rest(self.tokenizer)? {
            full_text.push_str(&rest);
            on_token(&rest);
        }

        let stats = GenStats {
            output_tokens,
            prompt_tokens,
            prefill_secs,
            decode_secs,
            ttft_ms,
            token_intervals_ms,
        };
        Ok((stats, full_text))
    }
}
