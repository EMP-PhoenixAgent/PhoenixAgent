//! Safetensors → GGUF-Content shim: **native F16/BF16 loading**.
//!
//! AmberCore's whole adapter surface reads models through candle's
//! `quantized_qwen3::Gguf` accessor (`tensor`, `qmatmul`, `rms_norm`,
//! `metadata`). Rather than write a parallel dense-weight path for every
//! architecture, loading a `.safetensors` model builds an **in-memory GGUF
//! `Content`** whose tensor entries point straight at the safetensors mmap:
//!
//! - Weights stay F16/BF16 **in place** (zero copies; mmap-backed), and
//!   `QMatMul` handles unquantized F16/BF16/F32 GGUF tensors natively — so
//!   every adapter works unchanged.
//! - A few HF tensors need real math before they match the GGUF layout
//!   (`A_log → −exp(A_log)` for qwen35). Those are materialized into a small
//!   in-memory **patch region** that lives *in front of* the file in the
//!   virtual address space (see [`PatchedReader`]).
//! - Hparams come from the sibling `config.json`, synthesized into the GGUF
//!   metadata table under the arch's usual keys (`qwen3.embedding_length`
//!   ← `hidden_size`, …) per arch map below.
//!
//! Supported: `qwen3` (+qwen3_moe names) and `qwen35` (text configs; the VL
//! vision tower tensors are ignored). Single-file safetensors only — sharded
//! repos need a multi-region reader (follow-up).

use crate::error::{Error, Result};
use candle_core::quantized::gguf_file::{Content, TensorInfo, Value, VersionedMagic};
use candle_core::quantized::GgmlDType;
use serde_json::Value as Json;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// One safetensors tensor from the header index.
#[derive(Clone)]
struct StEntry {
    dtype: String,
    shape: Vec<u64>,
    start: u64,
    end: u64,
}

/// A reader over a virtual address space laid out as
/// `[patch bytes | file 1 | file 2 | … | file N]` — the patch region is the
/// in-memory transform buffer; each file region is one shard of a (possibly
/// single-file) safetensors model. Reads and seeks route by offset into the
/// owning backing store; tensors never span regions.
pub struct PatchedReader {
    patch: Vec<u8>,
    /// (file, virtual base of its region, region length) in base order.
    files: Vec<(std::fs::File, u64, u64)>,
    /// Total virtual size = patch + all file regions.
    virt_end: u64,
    pos: u64,
}

impl PatchedReader {
    /// `files` are (handle, byte length) pairs; region bases are assigned
    /// after the patch, in order.
    fn new(patch: Vec<u8>, files: Vec<(std::fs::File, u64)>) -> Self {
        let mut base = patch.len() as u64;
        let mut regions = Vec::with_capacity(files.len());
        for (mut f, len) in files {
            let _ = f.seek(SeekFrom::Start(0));
            regions.push((f, base, len));
            base = base.saturating_add(len);
        }
        Self { patch, files: regions, virt_end: base, pos: 0 }
    }

}

impl Read for PatchedReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let patch_len = self.patch.len() as u64;
        if self.pos < patch_len {
            let n = (patch_len - self.pos).min(buf.len() as u64) as usize;
            let at = self.pos as usize;
            buf[..n].copy_from_slice(&self.patch[at..at + n]);
            self.pos += n as u64;
            return Ok(n);
        }
        let idx = self
            .files
            .iter()
            .position(|(_, base, len)| self.pos >= *base && self.pos < *base + *len)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "read past model regions")
            })?;
        let file_at = self.pos - self.files[idx].1;
        self.files[idx].0.seek(SeekFrom::Start(file_at))?;
        let n = self.files[idx].0.read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for PatchedReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let target = match pos {
            SeekFrom::Start(n) => n as i64,
            SeekFrom::Current(d) => self.pos as i64 + d,
            SeekFrom::End(d) => self.virt_end as i64 + d,
        };
        self.pos = target.max(0) as u64;
        Ok(self.pos)
    }
}

/// HF `model_type` → AmberCore arch string (the `general.architecture` value).
fn arch_from_model_type(mt: &str) -> Option<&'static str> {
    match mt {
        "qwen3" => Some("qwen3"),
        "qwen3_5" | "qwen3_5_text" => Some("qwen35"),
        _ => None,
    }
}

/// Parse the safetensors header: `(data_start, tensor index)`.
fn read_st_index(mut file: &std::fs::File) -> Result<(u64, HashMap<String, StEntry>)> {
    use std::io::Read as _;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| Error::Model(format!("safetensors header len: {e}")))?;
    let header_len = u64::from_le_bytes(len_buf);
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)
        .map_err(|e| Error::Model(format!("safetensors header: {e}")))?;
    let data_start = 8 + header_len;
    let json: Json = serde_json::from_slice(&header)
        .map_err(|e| Error::Model(format!("safetensors header json: {e}")))?;
    let obj = json.as_object().ok_or_else(|| Error::Model("bad safetensors header".into()))?;
    let mut out = HashMap::new();
    for (name, info) in obj {
        if name.starts_with("__") {
            continue; // __metadata__
        }
        let Some(info) = info.as_object() else { continue };
        let dtype = info.get("dtype").and_then(Json::as_str).unwrap_or_default().to_string();
        let shape: Vec<u64> = info
            .get("shape")
            .and_then(Json::as_array)
            .map(|a| a.iter().filter_map(Json::as_u64).collect())
            .unwrap_or_default();
        let offs = info
            .get("data_offsets")
            .and_then(Json::as_array)
            .map(|a| a.iter().filter_map(Json::as_u64).collect::<Vec<_>>())
            .unwrap_or_default();
        if offs.len() == 2 {
            out.insert(
                name.clone(),
                StEntry { dtype, shape, start: offs[0], end: offs[1] },
            );
        }
    }
    Ok((data_start, out))
}

/// IEEE 754 half → f32 (standard exponent/mantissa widening).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                // Subnormal: renormalize into f32.
                let mut e = -1i32;
                let mut f = frac as i32;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                sign | (((127 - 15 + e) as u32) << 23) | ((f & 0x3ff) as u32) << 13
            }
        }
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(bits)
}

fn map_dtype(dt: &str) -> Result<GgmlDType> {
    match dt {
        "F16" => Ok(GgmlDType::F16),
        "BF16" => Ok(GgmlDType::BF16),
        "F32" => Ok(GgmlDType::F32),
        other => Err(Error::Model(format!(
            "safetensors dtype `{other}` not supported by the F16 loader \
             (use an F16/BF16/F32 checkpoint)"
        ))),
    }
}

/// What to do for one GGUF tensor name.
enum MapKind {
    /// Zero-copy: point at the safetensors bytes (shape from the HF tensor,
    /// possibly squeezed/reshaped metadata-only).
    Copy { hf: &'static str, squeeze: bool },
    /// Materialize into the patch buffer as F32 after a transform.
    Transform { hf: &'static str, neg_exp_a_log: bool },
}

/// Per-arch HF→GGUF tensor name map (returns None → tensor absent).
fn tensor_map(arch: &str, gguf_name: &str) -> Option<MapKind> {
    // Non-layer tensors first — `blk.` stripping below would reject them.
    if !gguf_name.starts_with("blk.") {
        return match (arch, gguf_name) {
            (_, "token_embd.weight") => Some(MapKind::Copy { hf: "model.embed_tokens.weight", squeeze: false }),
            (_, "output_norm.weight") => Some(MapKind::Copy { hf: "model.norm.weight", squeeze: false }),
            (_, "output.weight") => Some(MapKind::Copy { hf: "lm_head.weight", squeeze: false }),
            _ => None,
        };
    }
    let n = gguf_name.strip_prefix("blk.")?.split('.').next()?;
    let rest = gguf_name.strip_prefix("blk.")?.strip_prefix(n)?;
    let rest = rest.strip_prefix('.')?;
    let pair: Option<MapKind> = match arch {
        "qwen3" => match rest {
            "attn_norm.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.input_layernorm.weight")), squeeze: false }),
            "ffn_norm.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.post_attention_layernorm.weight")), squeeze: false }),
            "attn_q.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.q_proj.weight")), squeeze: false }),
            "attn_k.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.k_proj.weight")), squeeze: false }),
            "attn_v.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.v_proj.weight")), squeeze: false }),
            "attn_output.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.o_proj.weight")), squeeze: false }),
            "attn_q_norm.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.q_norm.weight")), squeeze: false }),
            "attn_k_norm.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.k_norm.weight")), squeeze: false }),
            "ffn_gate.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.mlp.gate_proj.weight")), squeeze: false }),
            "ffn_up.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.mlp.up_proj.weight")), squeeze: false }),
            "ffn_down.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.mlp.down_proj.weight")), squeeze: false }),
            "token_embd.weight" => Some(MapKind::Copy { hf: "model.embed_tokens.weight", squeeze: false }),
            "output_norm.weight" => Some(MapKind::Copy { hf: "model.norm.weight", squeeze: false }),
            "output.weight" => Some(MapKind::Copy { hf: "lm_head.weight", squeeze: false }), // tied → absent, aliased by caller
            _ => None,
        },
        "qwen35" => match rest {
            "attn_norm.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.input_layernorm.weight")), squeeze: false }),
            "post_attention_norm.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.post_attention_layernorm.weight")), squeeze: false }),
            "ffn_gate.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.mlp.gate_proj.weight")), squeeze: false }),
            "ffn_up.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.mlp.up_proj.weight")), squeeze: false }),
            "ffn_down.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.mlp.down_proj.weight")), squeeze: false }),
            // GDN layers.
            "attn_qkv.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.linear_attn.in_proj_qkv.weight")), squeeze: false }),
            "attn_gate.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.linear_attn.in_proj_z.weight")), squeeze: false }),
            "ssm_beta.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.linear_attn.in_proj_b.weight")), squeeze: false }),
            "ssm_alpha.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.linear_attn.in_proj_a.weight")), squeeze: false }),
            "ssm_dt.bias" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.linear_attn.dt_bias")), squeeze: false }),
            "ssm_norm.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.linear_attn.norm.weight")), squeeze: false }),
            "ssm_out.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.linear_attn.out_proj.weight")), squeeze: false }),
            // (conv_dim, 1, k) → (conv_dim, k): metadata-only squeeze.
            "ssm_conv1d.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.linear_attn.conv1d.weight")), squeeze: true }),
            // A_log → −exp(A_log): the GGUF convention stores it pre-negated.
            "ssm_a" => Some(MapKind::Transform { hf: leak(format!("model.layers.{n}.linear_attn.A_log")), neg_exp_a_log: true }),
            // Full-attention layers (q_proj already carries the fused gate).
            "attn_q.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.q_proj.weight")), squeeze: false }),
            "attn_k.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.k_proj.weight")), squeeze: false }),
            "attn_v.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.v_proj.weight")), squeeze: false }),
            "attn_output.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.o_proj.weight")), squeeze: false }),
            "attn_q_norm.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.q_norm.weight")), squeeze: false }),
            "attn_k_norm.weight" => Some(MapKind::Copy { hf: leak(format!("model.layers.{n}.self_attn.k_norm.weight")), squeeze: false }),
            "token_embd.weight" => Some(MapKind::Copy { hf: "model.embed_tokens.weight", squeeze: false }),
            "output_norm.weight" => Some(MapKind::Copy { hf: "model.norm.weight", squeeze: false }),
            "output.weight" => Some(MapKind::Copy { hf: "lm_head.weight", squeeze: false }),
            _ => None,
        },
        _ => None,
    };
    pair
}

/// Tiny static-str cache so `Copy` can hold `&'static str` names built at
/// runtime (bounded: one entry per layer tensor actually loaded).
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// Synthesize the GGUF metadata table from `config.json` for the arch.
fn build_meta(arch: &str, cfg: &Json) -> Result<HashMap<String, Value>> {
    // Conditional-generation configs nest the text model under text_config.
    let text = cfg.get("text_config").unwrap_or(cfg);
    let get = |k: &str| text.get(k);
    let mut m = HashMap::new();
    m.insert("general.architecture".into(), Value::String(arch.into()));
    if let Some(n) = cfg.get("name").and_then(Json::as_str) {
        m.insert("general.name".into(), Value::String(n.into()));
    }
    let put_u = |m: &mut HashMap<String, Value>, k: &str, v: u32| {
        m.insert(k.into(), Value::U32(v));
    };
    let hidden = get("hidden_size").and_then(Json::as_u64).unwrap_or(1024) as u32;
    let layers = get("num_hidden_layers").and_then(Json::as_u64).unwrap_or(24) as u32;
    let heads = get("num_attention_heads").and_then(Json::as_u64).unwrap_or(32) as u32;
    let kv_heads = get("num_key_value_heads").and_then(Json::as_u64).unwrap_or(heads as u64) as u32;
    let head_dim = get("head_dim").and_then(Json::as_u64).unwrap_or((hidden / heads) as u64) as u32;
    let ffn = get("intermediate_size").and_then(Json::as_u64).unwrap_or(3584) as u32;
    let eps = get("rms_norm_eps").and_then(Json::as_f64).unwrap_or(1e-5) as f32;
    let ctx = get("max_position_embeddings").and_then(Json::as_u64).unwrap_or(4096) as u32;
    let eos = get("eos_token_id").and_then(Json::as_u64).unwrap_or(0) as u32;

    put_u(&mut m, &format!("{arch}.embedding_length"), hidden);
    put_u(&mut m, &format!("{arch}.block_count"), layers);
    put_u(&mut m, &format!("{arch}.feed_forward_length"), ffn);
    put_u(&mut m, &format!("{arch}.attention.head_count"), heads);
    put_u(&mut m, &format!("{arch}.attention.head_count_kv"), kv_heads);
    put_u(&mut m, &format!("{arch}.attention.key_length"), head_dim);
    put_u(&mut m, &format!("{arch}.attention.value_length"), head_dim);
    m.insert(format!("{arch}.attention.layer_norm_rms_epsilon"), Value::F32(eps));
    put_u(&mut m, &format!("{arch}.context_length"), ctx);
    put_u(&mut m, &format!("{arch}.eos_token_id"), eos);
    if let Some(bos) = get("bos_token_id").and_then(Json::as_u64) {
        put_u(&mut m, &format!("{arch}.bos_token_id"), bos as u32);
    }
    // rope: HF nests under rope_parameters in qwen3_5, flat elsewhere.
    let rope = text.get("rope_parameters").unwrap_or(text);
    let theta = rope.get("rope_theta").and_then(Json::as_f64).unwrap_or(1e6) as f32;
    m.insert(format!("{arch}.rope.freq_base"), Value::F32(theta));
    if arch == "qwen35" {
        // Partial rope: dimension_count = head_dim × partial_rotary_factor.
        let prf = rope.get("partial_rotary_factor").and_then(Json::as_f64).unwrap_or(0.25);
        put_u(&mut m, &format!("{arch}.rope.dimension_count"), (head_dim as f64 * prf) as u32);
        // GDN hparams (names follow the GGUF loader convention).
        let k_heads = get("linear_num_key_heads").and_then(Json::as_u64).unwrap_or(16) as u32;
        let v_heads = get("linear_num_value_heads").and_then(Json::as_u64).unwrap_or(16) as u32;
        let k_dim = get("linear_key_head_dim").and_then(Json::as_u64).unwrap_or(128) as u32;
        let v_dim = get("linear_value_head_dim").and_then(Json::as_u64).unwrap_or(128) as u32;
        let conv = get("linear_conv_kernel_dim").and_then(Json::as_u64).unwrap_or(4) as u32;
        let interval = get("full_attention_interval").and_then(Json::as_u64).unwrap_or(4) as u32;
        put_u(&mut m, &format!("{arch}.ssm.conv_kernel"), conv);
        put_u(&mut m, &format!("{arch}.ssm.state_size"), k_dim);
        put_u(&mut m, &format!("{arch}.ssm.group_count"), k_heads);
        put_u(&mut m, &format!("{arch}.ssm.time_step_rank"), v_heads);
        put_u(&mut m, &format!("{arch}.ssm.inner_size"), v_heads * v_dim);
        put_u(&mut m, &format!("{arch}.full_attention_interval"), interval);
        // layer_types → attention.recurrent_layers (true = recurrent).
        if let Some(types) = get("layer_types").and_then(Json::as_array) {
            let flags: Vec<Value> = types
                .iter()
                .map(|t| Value::Bool(t.as_str() == Some("linear_attention")))
                .collect();
            m.insert(format!("{arch}.attention.recurrent_layers"), Value::Array(flags));
        }
    }
    Ok(m)
}

/// The names an adapter will ask for, per arch — drives the Content build
/// (adapters fail loudly on missing tensors, so the list must be complete).
fn wanted_tensors(arch: &str, layers: usize) -> Vec<String> {
    let mut out = vec!["token_embd.weight".to_string(), "output_norm.weight".to_string()];
    for n in 0..layers {
        let p = format!("blk.{n}");
        match arch {
            "qwen3" => {
                for t in ["attn_norm", "ffn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
                          "attn_q_norm", "attn_k_norm",
                          "ffn_gate", "ffn_up", "ffn_down"] {
                    out.push(format!("{p}.{t}.weight"));
                }
            }
            "qwen35" => {
                for t in ["attn_norm", "post_attention_norm", "ffn_gate", "ffn_up", "ffn_down",
                          // GDN
                          "attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_conv1d", "ssm_out"] {
                    out.push(format!("{p}.{t}.weight"));
                }
                out.push(format!("{p}.ssm_dt.bias"));
                out.push(format!("{p}.ssm_a"));
                out.push(format!("{p}.ssm_norm.weight"));
                // Full attention (only present on full-attn layers — missing
                // HF names are skipped by the builder loop).
                for t in ["attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm", "attn_k_norm"] {
                    out.push(format!("{p}.{t}.weight"));
                }
            }
            _ => {}
        }
    }
    out
}

/// Build a GGUF `Content` + reader over a `.safetensors` file.
///
/// `config.json` must sit next to the model file. Returns the arch string,
/// the synthesized Content, and the patched reader the builder must use.
pub fn load_safetensors(
    path: &Path,
) -> Result<(String, Content, PatchedReader, HashMap<String, String>)> {
    let cfg_path = path.parent().unwrap_or(Path::new(".")).join("config.json");
    let cfg_raw = std::fs::read_to_string(&cfg_path).map_err(|e| {
        Error::Model(format!(
            "F16 safetensors models need a config.json next to them ({}): {e}",
            cfg_path.display()
        ))
    })?;
    let cfg: Json = serde_json::from_str(&cfg_raw)
        .map_err(|e| Error::Model(format!("parse {}: {e}", cfg_path.display())))?;
    // text_config carries the text model_type for conditional-generation repos.
    let mt = cfg
        .get("text_config")
        .and_then(|t| t.get("model_type"))
        .or_else(|| cfg.get("model_type"))
        .and_then(Json::as_str)
        .unwrap_or_default()
        .to_string();
    let arch = arch_from_model_type(&mt)
        .ok_or_else(|| {
            Error::Model(format!(
                "safetensors model_type `{mt}` has no F16 loader yet (supported: qwen3, qwen3_5)"
            ))
        })?
        .to_string();

    // Shard discovery: an index.json next to the entry file maps every
    // tensor to its shard. Without one, the entry is a plain single file.
    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let mut shard_names: Vec<String> = Vec::new();
    if let Ok(index_json) = std::fs::read_to_string(dir.join("index.json")) {
        if let Ok(map) = serde_json::from_str::<serde_json::Value>(&index_json) {
            if let Some(wm) = map.get("weight_map").and_then(|v| v.as_object()) {
                let mut names: Vec<String> = wm
                    .values()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect();
                names.sort();
                names.dedup();
                shard_names = names;
            }
        }
    }
    if shard_names.is_empty() {
        shard_names = vec![
            path.file_name().and_then(|n| n.to_str()).unwrap_or_default().to_string()
        ];
    }

    // Parse every shard's header; build name → (shard, entry) lookups.
    let mut shard_files = Vec::new();
    let mut shard_meta = Vec::new(); // (data_start, entries)
    for name in &shard_names {
        let spath = dir.join(name);
        let mut f = std::fs::File::open(&spath)
            .map_err(|e| Error::Model(format!("open shard {}: {e}", spath.display())))?;
        let (ds, mut entries) = read_st_index(&mut f)?;
        // Multimodal repos namespace the text model as
        // `model.language_model.*` (vision tower tensors are simply never
        // requested); alias them as plain `model.*` so the text name map
        // matches.
        const LM_PREFIX: &str = "model.language_model.";
        let aliases: Vec<(String, StEntry)> = entries
            .iter()
            .filter_map(|(k, v)| {
                let rest = k.strip_prefix(LM_PREFIX)?;
                let aliased = format!("model.{rest}");
                (!entries.contains_key(&aliased)).then_some((aliased, v.clone()))
            })
            .collect();
        entries.extend(aliases);
        let len = f.metadata().map_err(|e| Error::Model(e.to_string()))?.len();
        shard_files.push((f, len));
        shard_meta.push((ds, entries));
    }

    let meta = build_meta(&arch, &cfg)?;
    let layers = meta.get(&format!("{arch}.block_count")).and_then(|v| v.to_u32().ok()).unwrap_or(0) as usize;

    let mut patch: Vec<u8> = Vec::new();
    let mut tensor_infos: HashMap<String, TensorInfo> = HashMap::new();
    // Region bases: the patch comes first; shards stack after it in order.
    // Only final after pass 0 (transforms grow the patch), so pass 1 reads
    // them via this closure-free recomputation below.

    // Two passes: TRANSFORMS first, so the patch reaches its final size
    // before any zero-copy offset is computed (offsets bake in shard bases).
    let wanted = wanted_tensors(&arch, layers);
    let mut shard_bases: Vec<u64> = Vec::new();
    for pass in [0usize, 1] {
        if pass == 1 {
            let mut base = patch.len() as u64;
            shard_bases = shard_files
                .iter()
                .map(|(_, len)| {
                    let b = base;
                    base = base.saturating_add(*len);
                    b
                })
                .collect();
        }
    for gguf_name in &wanted {
        let Some(kind) = tensor_map(&arch, gguf_name) else { continue };
        let is_transform = matches!(kind, MapKind::Transform { .. });
        if (pass == 0) != is_transform {
            continue;
        }
        let hf_name = match &kind {
            MapKind::Copy { hf, .. } => hf.clone(),
            MapKind::Transform { hf, .. } => hf.clone(),
        };
        // Locate the tensor across shards (absent → layer-type-specific).
        let Some((shard, entry)) = shard_meta
            .iter()
            .enumerate()
            .find_map(|(si, (_, entries))| entries.get(hf_name).map(|e| (si, e)))
        else {
            continue;
        };
        let shard_data_start = shard_meta[shard].0;
        let gguf_name = gguf_name.clone();
        match &kind {
            MapKind::Copy { squeeze, .. } => {
                let mut ne: Vec<u64> = entry.shape.clone();
                if *squeeze {
                    ne.retain(|&d| d != 1);
                }
                // Content::read reverses GGUF ne order when parsing real
                // files; a directly-constructed Content stores candle
                // dims — which is exactly the HF layout (out, in).
                tensor_infos.insert(
                    gguf_name.clone(),
                    TensorInfo {
                        ggml_dtype: map_dtype(&entry.dtype)?,
                        shape: candle_core::Shape::from(
                            ne.iter().map(|&d| d as usize).collect::<Vec<_>>(),
                        ),
                        offset: shard_bases[shard] + shard_data_start + entry.start,
                    },
                );
            }
            MapKind::Transform { neg_exp_a_log, .. } => {
                if !*neg_exp_a_log {
                    continue; // no other transforms today
                }
                // Read the A_log values (F16 or F32 checkpoint) and append
                // −exp(A_log) as F32 to the patch.
                use std::io::Read;
                let mut buf = vec![0u8; (entry.end - entry.start) as usize];
                shard_files[shard].0
                    .seek(SeekFrom::Start(shard_data_start + entry.start))
                    .map_err(|e| Error::Model(e.to_string()))?;
                shard_files[shard].0
                    .read_exact(&mut buf)
                    .map_err(|e| Error::Model(format!("read {hf_name}: {e}")))?;
                let values: Vec<f32> = match entry.dtype.as_str() {
                    "F32" => buf
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                    "F16" => buf
                        .chunks_exact(2)
                        .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                        .collect(),
                    other => {
                        return Err(Error::Model(format!(
                            "A_log dtype {other} not supported (need F16/F32)"
                        )))
                    }
                };
                let n = values.len();
                let mut out = Vec::with_capacity(n * 4);
                for a_log in values {
                    let v = -a_log.exp();
                    out.extend_from_slice(&v.to_le_bytes());
                }
                let offset = patch.len() as u64;
                patch.extend_from_slice(&out);
                tensor_infos.insert(
                    gguf_name.clone(),
                    TensorInfo {
                        ggml_dtype: GgmlDType::F32,
                        shape: candle_core::Shape::from(vec![n]),
                        offset,
                    },
                );
            }
        }
    }
    }


    if tensor_infos.is_empty() {
        return Err(Error::Model(format!(
            "no mapped tensors in {} — is this a matching checkpoint?",
            path.display()
        )));
    }

    // Tied embeddings: alias `output` to the token table when lm_head is absent.
    let has_lm_head = shard_meta
        .iter()
        .any(|(_, entries)| entries.contains_key("lm_head.weight"));
    if !has_lm_head {
        if let Some(te) = tensor_infos.get("token_embd.weight") {
            let te = TensorInfo {
                ggml_dtype: te.ggml_dtype,
                shape: candle_core::Shape::from(te.shape.dims().to_vec()),
                offset: te.offset,
            };
            tensor_infos.insert("output.weight".into(), te);
        }
    }

    let content = Content {
        magic: VersionedMagic::GgufV3,
        metadata: meta.clone(),
        tensor_infos,
        tensor_data_offset: 0,
    };
    let reader = PatchedReader::new(patch, shard_files);

    // Surface the same convenience meta the GGUF loader does.
    let mut surf = HashMap::new();
    for key in [
        format!("{arch}.context_length"),
        format!("{arch}.eos_token_id"),
        format!("{arch}.block_count"),
        format!("{arch}.embedding_length"),
    ] {
        if let Some(v) = meta.get(&key) {
            let s: Option<String> = v
                .to_string()
                .ok()
                .map(|s| s.to_string())
                .or_else(|| v.to_u32().ok().map(|n| n.to_string()))
                .or_else(|| v.to_f32().ok().map(|n| n.to_string()));
            if let Some(s) = s {
                surf.insert(key, s);
            }
        }
    }

    tracing::info!(
        arch = %arch,
        tensors = content.tensor_infos.len(),
        "loaded safetensors via GGUF-content shim (F16 native)"
    );
    Ok((arch, content, reader, surf))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: a tiny qwen3 in F16 safetensors + config.json loads
    /// through the shim, builds via the registry, and generates — proving
    /// the zero-copy Content, the patch reader, and the config→meta
    /// synthesis all line up with the real GGUF path.
    #[test]
    fn qwen3_safetensors_end_to_end() {
        use candle_core::{DType, Device, Tensor};
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let vocab = 64usize;
        let hidden = 16usize;
        let layers = 2usize;
        let heads = 2usize;
        let kv = 1usize;
        let hd = hidden / heads;
        let ffn = 32usize;

        let f16 = |t: Tensor| -> Tensor { t.to_dtype(DType::F16).unwrap() };
        let mut tensors: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        tensors.insert("model.embed_tokens.weight".into(), f16(Tensor::randn(0f32, 1f32, (vocab, hidden), &dev).unwrap()));
        tensors.insert("model.norm.weight".into(), f16(Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
        for n in 0..layers {
            let p = format!("model.layers.{n}");
            tensors.insert(format!("{p}.input_layernorm.weight"), f16(Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            tensors.insert(format!("{p}.post_attention_layernorm.weight"), f16(Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            tensors.insert(format!("{p}.self_attn.q_proj.weight"), f16(Tensor::randn(0f32, 1f32, (heads * hd, hidden), &dev).unwrap()));
            tensors.insert(format!("{p}.self_attn.k_proj.weight"), f16(Tensor::randn(0f32, 1f32, (kv * hd, hidden), &dev).unwrap()));
            tensors.insert(format!("{p}.self_attn.v_proj.weight"), f16(Tensor::randn(0f32, 1f32, (kv * hd, hidden), &dev).unwrap()));
            tensors.insert(format!("{p}.self_attn.o_proj.weight"), f16(Tensor::randn(0f32, 1f32, (hidden, heads * hd), &dev).unwrap()));
            tensors.insert(format!("{p}.self_attn.q_norm.weight"), f16(Tensor::ones((hd,), DType::F32, &dev).unwrap()));
            tensors.insert(format!("{p}.self_attn.k_norm.weight"), f16(Tensor::ones((hd,), DType::F32, &dev).unwrap()));
            tensors.insert(format!("{p}.mlp.gate_proj.weight"), f16(Tensor::randn(0f32, 1f32, (ffn, hidden), &dev).unwrap()));
            tensors.insert(format!("{p}.mlp.up_proj.weight"), f16(Tensor::randn(0f32, 1f32, (ffn, hidden), &dev).unwrap()));
            tensors.insert(format!("{p}.mlp.down_proj.weight"), f16(Tensor::randn(0f32, 1f32, (hidden, ffn), &dev).unwrap()));
        }
        let model_path = dir.path().join("tiny-qwen3.safetensors");
        candle_core::safetensors::save(&tensors, &model_path).unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv,
            "head_dim": hd,
            "intermediate_size": ffn,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "eos_token_id": 7,
            "tie_word_embeddings": true,
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();

        let mut loaded = crate::model::gguf::LoadedModel::load(&model_path).unwrap();
        assert_eq!(loaded.arch, "qwen3");
        assert_eq!(
            crate::model::gguf::probe_arch(&model_path).unwrap(),
            "qwen3",
            "probe_arch reads config.json for safetensors"
        );
        let mut model = crate::model::registry::build(&mut loaded, &Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![3u32, 1, 4, 1], (1, 4), &Device::Cpu).unwrap();
        let logits = model.forward(&input, 0).unwrap();
        assert_eq!(logits.dims(), &[1, vocab]);
        let next = Tensor::from_vec(vec![5u32], (1, 1), &Device::Cpu).unwrap();
        let logits2 = model.forward(&next, 4).unwrap();
        assert_eq!(logits2.dims(), &[1, vocab]);
        model.clear_kv_cache();
        // F16 forward: logits are finite.
        let l = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(l.iter().all(|v| v.is_finite()), "finite logits");
    }

    /// Same dance for qwen35 — this one exercises the patch buffer (A_log →
    /// −exp), the conv1d squeeze, and the layer_types→recurrent_layers array.
    #[test]
    fn qwen35_safetensors_end_to_end() {
        use candle_core::{DType, Device, Tensor};
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let vocab = 64usize;
        let hidden = 16usize;
        let layers = 2usize;
        let heads = 2usize;
        let kv = 1usize;
        let hd = 8usize;
        let rope_dims = 4usize;
        let ffn = 32usize;
        let n_group = 2usize;
        let n_v = 2usize;
        let k_dim = 4usize;
        let head_v_dim = 4usize;
        let inner = n_v * head_v_dim;
        let conv_dim = inner + 2 * n_group * k_dim;

        let f16 = |t: Tensor| -> Tensor { t.to_dtype(DType::F16).unwrap() };
        let mut tensors: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        tensors.insert("model.embed_tokens.weight".into(), f16(Tensor::randn(0f32, 0.4f32, (vocab, hidden), &dev).unwrap()));
        tensors.insert("model.norm.weight".into(), f16(Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
        for n in 0..layers {
            let p = format!("model.layers.{n}");
            tensors.insert(format!("{p}.input_layernorm.weight"), f16(Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            tensors.insert(format!("{p}.post_attention_layernorm.weight"), f16(Tensor::ones((hidden,), DType::F32, &dev).unwrap()));
            tensors.insert(format!("{p}.mlp.gate_proj.weight"), f16(Tensor::randn(0f32, 0.4f32, (ffn, hidden), &dev).unwrap()));
            tensors.insert(format!("{p}.mlp.up_proj.weight"), f16(Tensor::randn(0f32, 0.4f32, (ffn, hidden), &dev).unwrap()));
            tensors.insert(format!("{p}.mlp.down_proj.weight"), f16(Tensor::randn(0f32, 0.4f32, (hidden, ffn), &dev).unwrap()));
            let la = format!("{p}.linear_attn");
            tensors.insert(format!("{la}.in_proj_qkv.weight"), f16(Tensor::randn(0f32, 0.4f32, (conv_dim, hidden), &dev).unwrap()));
            tensors.insert(format!("{la}.in_proj_z.weight"), f16(Tensor::randn(0f32, 0.4f32, (inner, hidden), &dev).unwrap()));
            tensors.insert(format!("{la}.in_proj_b.weight"), f16(Tensor::randn(0f32, 0.4f32, (n_v, hidden), &dev).unwrap()));
            tensors.insert(format!("{la}.in_proj_a.weight"), f16(Tensor::randn(0f32, 0.4f32, (n_v, hidden), &dev).unwrap()));
            tensors.insert(format!("{la}.dt_bias"), f16(Tensor::randn(0f32, 0.4f32, (n_v,), &dev).unwrap()));
            // A_log positive; the patch materializes −exp(A_log).
            tensors.insert(format!("{la}.A_log"), f16(Tensor::randn(0f32, 0.5f32, (n_v,), &dev).unwrap()));
            tensors.insert(format!("{la}.conv1d.weight"), f16(Tensor::randn(0f32, 0.4f32, (conv_dim, 1, 3), &dev).unwrap()));
            tensors.insert(format!("{la}.norm.weight"), f16(Tensor::ones((head_v_dim,), DType::F32, &dev).unwrap()));
            tensors.insert(format!("{la}.out_proj.weight"), f16(Tensor::randn(0f32, 0.4f32, (hidden, inner), &dev).unwrap()));
            if n == 1 {
                // Full-attention layer (layer_types[1] = full_attention).
                let sa = format!("{p}.self_attn");
                tensors.insert(format!("{sa}.q_proj.weight"), f16(Tensor::randn(0f32, 0.4f32, (heads * 2 * hd, hidden), &dev).unwrap()));
                tensors.insert(format!("{sa}.k_proj.weight"), f16(Tensor::randn(0f32, 0.4f32, (kv * hd, hidden), &dev).unwrap()));
                tensors.insert(format!("{sa}.v_proj.weight"), f16(Tensor::randn(0f32, 0.4f32, (kv * hd, hidden), &dev).unwrap()));
                tensors.insert(format!("{sa}.o_proj.weight"), f16(Tensor::randn(0f32, 0.4f32, (hidden, heads * hd), &dev).unwrap()));
                tensors.insert(format!("{sa}.q_norm.weight"), f16(Tensor::ones((hd,), DType::F32, &dev).unwrap()));
                tensors.insert(format!("{sa}.k_norm.weight"), f16(Tensor::ones((hd,), DType::F32, &dev).unwrap()));
            }
        }
        let model_path = dir.path().join("tiny-qwen35.safetensors");
        candle_core::safetensors::save(&tensors, &model_path).unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv,
            "head_dim": hd,
            "intermediate_size": ffn,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 128,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 3,
            "linear_num_key_heads": n_group,
            "linear_key_head_dim": k_dim,
            "linear_num_value_heads": n_v,
            "linear_value_head_dim": head_v_dim,
            "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.5 },
            "eos_token_id": 7,
            "tie_word_embeddings": true,
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();

        let mut loaded = crate::model::gguf::LoadedModel::load(&model_path).unwrap();
        assert_eq!(loaded.arch, "qwen35");
        let mut model = crate::model::registry::build(&mut loaded, &Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![3u32, 1, 4, 1], (1, 4), &Device::Cpu).unwrap();
        let logits = model.forward(&input, 0).unwrap();
        assert_eq!(logits.dims(), &[1, vocab]);
        let next = Tensor::from_vec(vec![5u32], (1, 1), &Device::Cpu).unwrap();
        let logits2 = model.forward(&next, 4).unwrap();
        assert_eq!(logits2.dims(), &[1, vocab]);
        model.clear_kv_cache();
        // F16 forward: logits are finite.
        let l = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(l.iter().all(|v| v.is_finite()), "finite logits");
    }

    /// Sharded repo: tensors split across two safetensors files + an
    /// index.json weight map — loading the FIRST shard must see the whole
    /// model through the multi-region reader.
    #[test]
    fn qwen3_sharded_safetensors_end_to_end() {
        use candle_core::{DType, Device, Tensor};
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().unwrap();
        let vocab = 64usize;
        let hidden = 16usize;
        let layers = 2usize;
        let heads = 2usize;
        let kv = 1usize;
        let hd = hidden / heads;
        let ffn = 32usize;

        let f16 = |t: Tensor| -> Tensor { t.to_dtype(DType::F16).unwrap() };
        // Shard 1: embeddings + layer 0; shard 2: layer 1 + norm.
        let mut s1: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        let mut s2: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        let mut weight_map = serde_json::Map::new();
        let mut put = |shard: &mut std::collections::HashMap<String, Tensor>,
                       name: String,
                       t: Tensor,
                       file: &str| {
            weight_map.insert(name.clone(), serde_json::Value::String(file.to_string()));
            shard.insert(name, f16(t));
        };
        put(&mut s1, "model.embed_tokens.weight".into(),
            Tensor::randn(0f32, 1f32, (vocab, hidden), &dev).unwrap(), "model-00001-of-00002.safetensors");
        put(&mut s2, "model.norm.weight".into(),
            Tensor::ones((hidden,), DType::F32, &dev).unwrap(), "model-00002-of-00002.safetensors");
        for n in 0..layers {
            let (shard, file) = if n == 0 {
                (&mut s1, "model-00001-of-00002.safetensors")
            } else {
                (&mut s2, "model-00002-of-00002.safetensors")
            };
            let p = format!("model.layers.{n}");
            put(shard, format!("{p}.input_layernorm.weight"), Tensor::ones((hidden,), DType::F32, &dev).unwrap(), file);
            put(shard, format!("{p}.post_attention_layernorm.weight"), Tensor::ones((hidden,), DType::F32, &dev).unwrap(), file);
            put(shard, format!("{p}.self_attn.q_proj.weight"), Tensor::randn(0f32, 1f32, (heads * hd, hidden), &dev).unwrap(), file);
            put(shard, format!("{p}.self_attn.k_proj.weight"), Tensor::randn(0f32, 1f32, (kv * hd, hidden), &dev).unwrap(), file);
            put(shard, format!("{p}.self_attn.v_proj.weight"), Tensor::randn(0f32, 1f32, (kv * hd, hidden), &dev).unwrap(), file);
            put(shard, format!("{p}.self_attn.o_proj.weight"), Tensor::randn(0f32, 1f32, (hidden, heads * hd), &dev).unwrap(), file);
            put(shard, format!("{p}.self_attn.q_norm.weight"), Tensor::ones((hd,), DType::F32, &dev).unwrap(), file);
            put(shard, format!("{p}.self_attn.k_norm.weight"), Tensor::ones((hd,), DType::F32, &dev).unwrap(), file);
            put(shard, format!("{p}.mlp.gate_proj.weight"), Tensor::randn(0f32, 1f32, (ffn, hidden), &dev).unwrap(), file);
            put(shard, format!("{p}.mlp.up_proj.weight"), Tensor::randn(0f32, 1f32, (ffn, hidden), &dev).unwrap(), file);
            put(shard, format!("{p}.mlp.down_proj.weight"), Tensor::randn(0f32, 1f32, (hidden, ffn), &dev).unwrap(), file);
        }
        candle_core::safetensors::save(&s1, dir.path().join("model-00001-of-00002.safetensors")).unwrap();
        candle_core::safetensors::save(&s2, dir.path().join("model-00002-of-00002.safetensors")).unwrap();
        let index_json = serde_json::json!({
            "metadata": { "total_size": 12345 },
            "weight_map": weight_map,
        });
        std::fs::write(dir.path().join("index.json"), index_json.to_string()).unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": hidden, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": kv,
            "head_dim": hd, "intermediate_size": ffn, "rms_norm_eps": 1e-5,
            "max_position_embeddings": 128, "rope_theta": 10000.0,
            "eos_token_id": 7, "tie_word_embeddings": true,
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();

        // Entry = the FIRST shard only.
        let entry = dir.path().join("model-00001-of-00002.safetensors");
        let mut loaded = crate::model::gguf::LoadedModel::load(&entry).unwrap();
        assert_eq!(loaded.arch, "qwen3");
        let mut model = crate::model::registry::build(&mut loaded, &Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![3u32, 1, 4, 1], (1, 4), &Device::Cpu).unwrap();
        let logits = model.forward(&input, 0).unwrap();
        assert_eq!(logits.dims(), &[1, vocab]);
        let next = Tensor::from_vec(vec![5u32], (1, 1), &Device::Cpu).unwrap();
        assert_eq!(model.forward(&next, 4).unwrap().dims(), &[1, vocab]);
        let l = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(l.iter().all(|v| v.is_finite()), "finite logits across shards");
    }

    #[test]
    fn patchedReader_routes_by_offset() {
        let patch = b"PATCH".to_vec();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tail.bin");
        std::fs::write(&p, b"TAILDATA").unwrap();
        let f = std::fs::File::open(&p).unwrap();
        // File region maps at virtual offset 5.
        let mut r = PatchedReader::new(patch, vec![(f, 8)]);
        let mut buf = [0u8; 5];
        r.seek(SeekFrom::Start(0)).unwrap();
        r.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"PATCH");
        r.seek(SeekFrom::Start(5)).unwrap();
        r.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"TAILD");
    }
}
