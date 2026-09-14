//! Shared building blocks for the GGUF architecture adapters.
//!
//! The per-family modules ([`crate::model::granite`], [`crate::model::deepseek2`],
//! ...) each own their compute graph, but three things are family-independent:
//!
//! - [`causal_mask`] — the additive `(b, 1, tgt, tgt + offset)` mask with an
//!   optional sliding window (gemma4 / granite_swa local layers).
//! - [`Meta`] — typed reads from a GGUF metadata table under an architecture
//!   namespace (`{arch}.attention.head_count`, ...), with llama.cpp-compatible
//!   defaults. Hand-rolled instead of serde so unknown-key tolerance and
//!   per-key defaults stay explicit.
//! - [`gguf_test_file`] — a test-only helper that writes a real (tiny) GGUF to
//!   disk from QTensors + metadata so adapters can be exercised end-to-end
//!   (`load → build → forward`) without any model download.

use candle_core::quantized::gguf_file::Value;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

/// Build an additive causal attention mask of shape `(b, 1, tgt, tgt + offset)`
/// in `dtype` on `device`. `offset` is the absolute position of the first token
/// in the current chunk (the KV-cache length). `sw`, when set, additionally
/// masks keys outside a sliding window of that many tokens around each query —
/// the local-attention layers of gemma4 / granite_swa.
pub fn causal_mask(
    device: &Device,
    dtype: DType,
    b: usize,
    tgt: usize,
    offset: usize,
    sw: Option<usize>,
) -> candle_core::Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<_> = (0..tgt)
        .flat_map(|i| {
            (0..(tgt + offset)).map(move |j| {
                let past_ok = j <= i + offset;
                let sw_ok = match sw {
                    Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                    None => true,
                };
                if past_ok && sw_ok {
                    0.
                } else {
                    minf
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), device)?.to_dtype(dtype)
}

/// Typed views over a parsed GGUF metadata table, namespaced by architecture.
///
/// GGUF keys look like `granite.attention.head_count`; [`Meta::req_u32`] reads
/// `"{arch}.attention.head_count"` and errors (with the key in the message) if
/// it is missing, while the `opt_*` readers take a default.
pub struct Meta<'a> {
    map: &'a HashMap<String, Value>,
    arch: &'a str,
}

impl<'a> Meta<'a> {
    pub fn new(map: &'a HashMap<String, Value>, arch: &'a str) -> Self {
        Self { map, arch }
    }

    fn get(&self, key: &str) -> Option<&Value> {
        self.map.get(&format!("{}.{}", self.arch, key))
    }

    /// Read a required u32-class integer (`to_u32` accepts u16/u32/i32/... too).
    pub fn req_u32(&self, key: &str) -> candle_core::Result<usize> {
        match self.get(key) {
            None => candle_core::bail!("missing metadata key {}.{}", self.arch, key),
            Some(v) => Ok(v.to_u32()? as usize),
        }
    }

    /// Read an optional u32-class integer, falling back to `default`.
    pub fn opt_u32(&self, key: &str, default: usize) -> usize {
        self.get(key).and_then(|v| v.to_u32().ok()).map(|n| n as usize).unwrap_or(default)
    }

    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Read an optional f32 (`to_f32` also accepts f64).
    pub fn opt_f32(&self, key: &str, default: f32) -> f32 {
        self.get(key).and_then(|v| v.to_f32().ok()).unwrap_or(default)
    }

    /// Read an optional bool.
    pub fn opt_bool(&self, key: &str, default: bool) -> bool {
        self.get(key).and_then(|v| v.to_bool().ok()).unwrap_or(default)
    }

    /// Read an optional array of u32 (`attention.sliding_window_pattern` and
    /// friends are written as bool arrays by llama.cpp; both spellings are
    /// accepted). Returns one entry per layer when present.
    pub fn opt_bool_array(&self, key: &str, len: usize) -> Option<Vec<bool>> {
        match self.get(key) {
            Some(Value::Array(vs)) => Some(
                vs.iter()
                    .map(|v| match v {
                        Value::Bool(b) => *b,
                        Value::U32(n) => *n != 0,
                        Value::U8(n) => *n != 0,
                        Value::I32(n) => *n != 0,
                        _ => false,
                    })
                    .collect::<Vec<bool>>(),
            ),
            // A bare integer is the gemma "5:1" ratio in llama.cpp's
            // get_key_or_arr convention: layer i is global (non-sliding) when
            // (i + 1) % n == 0, sliding otherwise.
            Some(Value::U32(n)) if *n > 0 => {
                let n = *n as usize;
                Some((0..len).map(|i| (i + 1) % n != 0).collect())
            }
            _ => None,
        }
        .map(|v| {
            if v.len() == len {
                v
            } else {
                // Length mismatch: repeat/truncate defensively.
                (0..len).map(|i| v.get(i % v.len()).copied().unwrap_or(false)).collect()
            }
        })
    }

    /// Read an optional string (e.g. `rope.scaling.type`).
    pub fn opt_str(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|v| v.to_string().ok().map(|s| s.to_string()))
    }
}

/// Base RoPE inverse frequencies `1 / base^(2i/d)` for `dim` rotary dims.
fn base_inv_freq(base: f64, dim: usize) -> Vec<f32> {
    (0..dim).step_by(2).map(|i| 1f32 / base.powf(i as f64 / dim as f64) as f32).collect()
}

/// llama.cpp-style RoPE frequency scaling for long-context models.
///
/// - `Linear`: every frequency divided by `factor` (the original NTK-x trick).
/// - `YaRN`/`llama3`: wavelengths shorter than `high_wavelen` keep their
///   frequency, longer than `low_wavelen` get the full `factor`, between
///   the two they blend smoothly — the same curve llama.cpp's `rope_yarn`
///   applies (modulo its out-of-range extrapolation ramp, which only matters
///   beyond `original_context_length`).
///
/// The thresholds are `orig_ctx / low` and `orig_ctx / high` in both
/// spellings: llama3-style GGUFs carry `low_freq_factor`/`high_freq_factor`
/// directly, YaRN GGUFs (DeepSeek) carry `beta_slow`/`beta_fast` which play
/// the same roles (slow → low band, fast → high band).
pub fn scaled_inv_freq(
    base: f64,
    dim: usize,
    scaling: Scaling,
) -> Vec<f32> {
    match scaling {
        Scaling::None => base_inv_freq(base, dim),
        Scaling::Linear { factor } => base_inv_freq(base, dim)
            .into_iter()
            .map(|f| f / factor as f32)
            .collect(),
        Scaling::Yarn { factor, original_ctx, low_freq_factor, high_freq_factor } => {
            let low_wavelen = original_ctx as f32 / low_freq_factor;
            let high_wavelen = original_ctx as f32 / high_freq_factor;
            base_inv_freq(base, dim)
                .into_iter()
                .map(|freq| {
                    let wavelen = 2.0 * std::f32::consts::PI / freq;
                    if wavelen < high_wavelen {
                        freq
                    } else if wavelen > low_wavelen {
                        freq / factor as f32
                    } else {
                        let smooth = (original_ctx as f32 / wavelen - low_freq_factor)
                            / (high_freq_factor - low_freq_factor);
                        (1.0 - smooth) * freq / factor as f32 + smooth * freq
                    }
                })
                .collect()
        }
    }
}

/// Which RoPE scaling a model trained with (parsed from
/// `{arch}.rope.scaling.type` + friends by [`parse_scaling`]).
#[derive(Debug, Clone, Copy)]
pub enum Scaling {
    None,
    Linear { factor: f32 },
    Yarn { factor: f32, original_ctx: usize, low_freq_factor: f32, high_freq_factor: f32 },
}

/// Parse `{arch}.rope.scaling.*` metadata into a [`Scaling`]. Explicit
/// llama3-style factors win when present; otherwise the YaRN betas fill the
/// same slots with the YaRN reference defaults (slow 1, fast 32).
pub fn parse_scaling(meta: &Meta) -> Scaling {
    let kind = meta.opt_str("rope.scaling.type").unwrap_or_default();
    let factor = meta.opt_f32("rope.scaling.factor", 1.0);
    if factor == 1.0 || kind.is_empty() {
        return Scaling::None;
    }
    let original_ctx = meta.opt_u32("rope.scaling.original_context_length", 4096);
    let low = match meta_f32_if_present(meta, "rope.scaling.low_freq_factor") {
        Some(v) => v,
        None => meta.opt_f32("rope.scaling.beta_slow", 1.0),
    };
    let high = match meta_f32_if_present(meta, "rope.scaling.high_freq_factor") {
        Some(v) => v,
        None => meta.opt_f32("rope.scaling.beta_fast", 32.0),
    };
    match kind.as_str() {
        "linear" => Scaling::Linear { factor },
        // yarn and llama3 use the same smooth blend in practice
        _ => Scaling::Yarn { factor, original_ctx, low_freq_factor: low, high_freq_factor: high },
    }
}

/// `opt_f32` with a NaN sentinel default can't distinguish "absent" from a
/// stored NaN; helper for keys where absence changes the fallback chain.
fn meta_f32_if_present(meta: &Meta, key: &str) -> Option<f32> {
    let v = meta.opt_f32(key, f32::NAN);
    (!v.is_nan()).then_some(v)
}

/// A per-layer KV cache slot for architectures whose attention layers are
/// sparse (hybrids) or shared: stores K/V as `(b, heads, t, dim)` and grows by
/// concatenation. `append` returns the full cache (cloned) so the caller can
/// attend over the whole history; `ConcatKvCache` can't express read-without-
/// append or per-layer gaps, which is why this exists.
///
/// Lives in `common` since both gemma4 (cross-layer KV sharing) and qwen35
/// (3-of-4 recurrent layers keep no KV at all) need it.
#[derive(Default)]
pub struct KvSlot {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvSlot {
    /// Append this step's K/V and return the full cache contents.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        let k_all = match &self.k {
            Some(prev) => Tensor::cat(&[prev, k], 2)?.contiguous()?,
            None => k.clone(),
        };
        let v_all = match &self.v {
            Some(prev) => Tensor::cat(&[prev, v], 2)?.contiguous()?,
            None => v.clone(),
        };
        self.k = Some(k_all.clone());
        self.v = Some(v_all.clone());
        Ok((k_all, v_all))
    }

    /// The current cache contents (owned clones; `repeat_kv` takes tensors
    /// by value).
    pub fn current(&self) -> candle_core::Result<(Tensor, Tensor)> {
        match (&self.k, &self.v) {
            (Some(k), Some(v)) => Ok((k.clone(), v.clone())),
            _ => candle_core::bail!("KV slot read before first write"),
        }
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// Precomputed cos/sin tables for one rope convention, `(max_pos, dim/2)` each
/// in `dtype` — shared by the adapters that apply rope themselves.
pub struct RopeTables {
    pub cos: Tensor,
    pub sin: Tensor,
}

impl RopeTables {
    /// Build tables for `dim` rotary dims over `max_pos` positions with the
    /// given (possibly scaled) inverse frequencies.
    pub fn new(inv_freq: &[f32], max_pos: usize, dtype: DType, device: &Device) -> candle_core::Result<Self> {
        let inv = Tensor::from_slice(inv_freq, (1, inv_freq.len()), device)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_pos as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_pos, 1))?;
        let freqs = t.matmul(&inv)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    /// Apply **interleaved** (llama-family / ggml `NORM`) rope to
    /// `(b, heads, seq, dim)`, rotating pairs `(2i, 2i+1)`. The GGUF
    /// conversion pre-permutes llama-family q/k weights into this layout.
    pub fn apply_interleaved(
        &self,
        x: &Tensor,
        dtype: DType,
        pos: usize,
    ) -> candle_core::Result<Tensor> {
        let seq = x.dims()[2];
        let cos = self.cos.narrow(0, pos, seq)?.to_dtype(dtype)?;
        let sin = self.sin.narrow(0, pos, seq)?.to_dtype(dtype)?;
        candle_nn::rotary_emb::rope_i(x, &cos, &sin)
    }

    /// Apply **half-split** (NeoX / ggml `NEOX`) rope to `(b, heads, seq, dim)`,
    /// pairing `i` with `i + dim/2` — the gemma/qwen/deepseek convention.
    pub fn apply_half(
        &self,
        x: &Tensor,
        dtype: DType,
        pos: usize,
    ) -> candle_core::Result<Tensor> {
        let seq = x.dims()[2];
        let cos = self.cos.narrow(0, pos, seq)?.to_dtype(dtype)?;
        let sin = self.sin.narrow(0, pos, seq)?.to_dtype(dtype)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)
    }
}

/// Write a minimal-but-valid GGUF file from metadata + quantized tensors.
///
/// Test-only (each family's tests builds a tiny synthetic model, loads it
/// through [`crate::model::gguf::LoadedModel`] and the registry, and runs a
/// forward pass). Tensors quantize to Q8_0 where the 32-element block layout
/// fits and stay F32 otherwise (norms, router biases — exactly what real
/// llama.cpp conversions do). Callers pass Rust-shaped tensors in the same
/// orientation the real files use **after** candle's dim reversal, i.e.
/// `token_embd` is `(vocab, hidden)`, projections are `(in, out)`.
#[cfg(test)]
pub(crate) fn gguf_test_file(
    path: &std::path::Path,
    metadata: &[(&str, Value)],
    tensors: &[(String, Tensor)],
) -> candle_core::Result<()> {
    use candle_core::quantized::gguf_file;
    use candle_core::quantized::QTensor;
    use std::io::BufWriter;

    let quantized: Vec<QTensor> = tensors
        .iter()
        .map(|(_, t)| {
            let last = t.dims().last().copied().unwrap_or(0);
            let dtype = if last % 32 == 0 {
                candle_core::quantized::GgmlDType::Q8_0
            } else {
                candle_core::quantized::GgmlDType::F32
            };
            QTensor::quantize(t, dtype)
        })
        .collect::<candle_core::Result<_>>()?;
    let metadata_refs: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors
        .iter()
        .zip(quantized.iter())
        .map(|((name, _), qt)| (name.as_str(), qt))
        .collect();

    let file = std::fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    gguf_file::write(&mut w, &metadata_refs, &tensor_refs)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_inv_freq_yarn_blends_low_and_high() {
        // 8 rope dims, base 10_000, factor 40, ctx 4096.
        let plain = scaled_inv_freq(10_000.0, 8, Scaling::None);
        let yarn = scaled_inv_freq(
            10_000.0,
            8,
            Scaling::Yarn {
                factor: 40.0,
                original_ctx: 4096,
                low_freq_factor: 1.0,  // beta_slow = 1  → low_wavelen 4096
                high_freq_factor: 32.0, // beta_fast = 32 → high_wavelen 128
            },
        );
        // Highest frequency (i=0): wavelen < high_wavelen → untouched.
        assert!((plain[0] - yarn[0]).abs() < 1e-6, "high freqs pass through");
        // Lowest frequencies: wavelen > low_wavelen → divided by factor.
        let last = *plain.last().unwrap();
        let scaled_last = *yarn.last().unwrap();
        assert!((scaled_last - last / 40.0).abs() < 1e-6, "low freqs scaled by factor");

        let linear = scaled_inv_freq(10_000.0, 8, Scaling::Linear { factor: 4.0 });
        assert!((linear[0] - plain[0] / 4.0).abs() < 1e-6);
    }

    #[test]
    fn parse_scaling_reads_both_spelling_sets() {
        let mut map = HashMap::new();
        map.insert("ds.rope.scaling.type".to_string(), Value::String("yarn".into()));
        map.insert("ds.rope.scaling.factor".to_string(), Value::F32(40.0));
        map.insert("ds.rope.scaling.original_context_length".to_string(), Value::U32(4096));
        map.insert("ds.rope.scaling.beta_fast".to_string(), Value::F32(32.0));
        map.insert("ds.rope.scaling.beta_slow".to_string(), Value::F32(1.0));
        let meta = Meta::new(&map, "ds");
        match parse_scaling(&meta) {
            Scaling::Yarn { factor, original_ctx, low_freq_factor, high_freq_factor } => {
                assert_eq!(factor, 40.0);
                assert_eq!(original_ctx, 4096);
                assert!((low_freq_factor - 1.0).abs() < 1e-5, "beta_slow=1 → low=1");
                assert!((high_freq_factor - 32.0).abs() < 1e-5, "beta_fast=32 → high=32");
            }
            other => panic!("expected Yarn, got {other:?}"),
        }

        map.insert("g.rope.scaling.type".to_string(), Value::String("linear".into()));
        map.insert("g.rope.scaling.factor".to_string(), Value::F32(8.0));
        assert!(matches!(parse_scaling(&Meta::new(&map, "g")), Scaling::Linear { factor: 8.0 }));

        // No scaling keys → None.
        assert!(matches!(parse_scaling(&Meta::new(&map, "other")), Scaling::None));
    }

    #[test]
    fn causal_mask_blocks_future_and_out_of_window_keys() {
        let dev = Device::Cpu;
        let m = causal_mask(&dev, DType::F32, 1, 3, 2, None).unwrap();
        assert_eq!(m.dims(), &[1, 1, 3, 5]);
        let v = m.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Key 4 > query 0 + offset 2 → masked.
        assert!(v[4].is_infinite(), "future keys masked");
        // Key 0 is visible to query 2 (position 4 sees 0..=4).
        assert!(v[10] == 0.0, "past keys visible");

        // Sliding window of 1: query at absolute 4 may only see keys 3..=4.
        let sw = causal_mask(&dev, DType::F32, 1, 3, 2, Some(1)).unwrap();
        let v = sw.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Row for query index 2 (absolute 4): keys 0..=4, window 1 → only 3,4 open.
        for (j, val) in v[10..15].iter().enumerate() {
            let expected_open = j == 3 || j == 4;
            assert_eq!(val.is_finite(), expected_open, "query2 key{j}");
        }
    }

    #[test]
    fn meta_reads_namespaced_keys_with_defaults() {
        let mut map = HashMap::new();
        map.insert("granite.attention.head_count".to_string(), Value::U32(32));
        map.insert("granite.attention.sliding_window_pattern".to_string(), Value::Bool(true));
        map.insert(
            "deepseek2.rope.scaling.type".to_string(),
            Value::String("yarn".into()),
        );
        let meta = Meta::new(&map, "granite");
        assert_eq!(meta.req_u32("attention.head_count").unwrap(), 32);
        assert_eq!(meta.opt_u32("attention.head_count_kv", 8), 8, "absent → default");
        assert!(meta.req_u32("attention.head_count_kv").is_err(), "required key errors");
        assert!(meta.has("attention.sliding_window_pattern"));
        assert!(!meta.has("rope.freq_base"));
        let ds = Meta::new(&map, "deepseek2");
        assert_eq!(ds.opt_str("rope.scaling.type").as_deref(), Some("yarn"));
    }

    #[test]
    fn meta_bool_array_accepts_llama_cpp_spellings() {
        let mut map = HashMap::new();
        map.insert(
            "gemma4.attention.sliding_window_pattern".to_string(),
            Value::Array(vec![Value::Bool(false), Value::Bool(true), Value::Bool(true)]),
        );
        map.insert(
            "x.attention.pattern".to_string(),
            Value::Array(vec![Value::U32(0), Value::U32(1)]),
        );
        let meta = Meta::new(&map, "gemma4");
        assert_eq!(
            meta.opt_bool_array("attention.sliding_window_pattern", 3).unwrap(),
            vec![false, true, true]
        );
        // Repeats when the array is shorter than the layer count.
        let meta = Meta::new(&map, "x");
        assert_eq!(meta.opt_bool_array("attention.pattern", 4).unwrap(), vec![false, true, false, true]);
    }
}
