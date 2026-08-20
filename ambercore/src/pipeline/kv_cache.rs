//! KV (key/value) cache management.
//!
//! During autoregressive generation, each transformer layer produces key and
//! value tensors for the tokens seen so far. Recomputing them every step is
//! O(n²); caching them makes decoding O(1) per step. This module owns the cache
//! shape and the append/eviction logic.
//!
//! M1 implements the basic append-only cache. M5+ may add paged caches (à la
//! vLLM) for continuous batching, which is where the real efficiency gains live.

use crate::error::Result;

/// An append-only KV cache for one generation session.
///
/// Typed placeholder for the candle-backed cache (`Vec<(Tensor, Tensor)>` per
/// layer). M1 fills in the real append + slicing logic.
#[derive(Debug, Default)]
pub struct KvCache {
    /// Number of tokens currently cached (sequence position).
    pub len: usize,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append the K/V tensors for `n` new tokens. M1.
    pub fn append(&mut self, _n: usize) -> Result<()> {
        self.len += _n;
        Ok(())
    }

    /// Reset the cache for a new session.
    pub fn clear(&mut self) {
        self.len = 0;
    }
}
