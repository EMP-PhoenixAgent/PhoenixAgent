//! GGUF file loading.
//!
//! Wraps [`candle::quantized::gguf_file`], which parses the GGUF container,
//! reads its metadata key/value table, and exposes the quantized tensor blobs
//! (lazily, via `Content::tensor`). A [`LoadedModel`] owns the open file handle
//! + parsed [`Content`] so a downstream architecture constructor can call
//! `ModelWeights::from_gguf(content, &mut file, device)`.
//!
//! Metadata of interest read at load time:
//! - `general.architecture` — selects the architecture (e.g. `"qwen2"`).
//! - `general.name` — human-readable model name.
//! - `<arch>.context_length` — max sequence length.
//! - the EOS token id, looked up via `<arch>.eos_token_id` with a fallback to
//!   the tokenizer's `<|endoftext|>` / `<|im_end|>` at the pipeline level.

use crate::error::{Error, Result};
use candle_core::quantized::gguf_file::Content;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::path::Path;

/// The model file the architecture builder reads tensors from: a plain GGUF,
/// or the safetensors-backed patched reader (see [`crate::model::st_shim`]).
/// Adapters are generic over `Read + Seek`, so both flow through `Gguf::new`
/// unchanged.
pub enum ModelFile {
    Gguf(File),
    Safetensors(crate::model::st_shim::PatchedReader),
}

impl Read for ModelFile {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            ModelFile::Gguf(f) => f.read(buf),
            ModelFile::Safetensors(r) => r.read(buf),
        }
    }
}

impl Seek for ModelFile {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
        match self {
            ModelFile::Gguf(f) => f.seek(pos),
            ModelFile::Safetensors(r) => r.seek(pos),
        }
    }
}

/// A loaded model, ready to be handed to an architecture constructor.
///
/// Owns the open file (so tensors can be read lazily from disk by the
/// architecture builder) plus the parsed [`Content`] (consumed by `from_gguf`
/// — for safetensors models the Content is synthesized by [`crate::model::st_shim`]
/// over the mmap, keeping weights F16/BF16 in place).
pub struct LoadedModel {
    /// Architecture string from `general.architecture` (e.g. `"qwen2"`, `"qwen35"`).
    pub arch: String,
    /// Human-readable model name from `general.name` (best-effort).
    pub name: Option<String>,
    /// The parsed GGUF content — handed to `ModelWeights::from_gguf`. It is an
    /// `Option` because the architecture builder consumes it.
    pub content: Option<Content>,
    /// The still-open model file, kept so the builder can read tensor data.
    pub file: ModelFile,
    /// Selected metadata values surfaced for convenience (max seq len, etc.).
    pub meta: HashMap<String, String>,
}

/// Read a model's architecture string: the GGUF header's
/// `general.architecture`, or — for `.safetensors` — the `model_type` of the
/// sibling `config.json` mapped onto the same arch names.
///
/// Cheap regardless of model size. Used to validate a freshly downloaded
/// model against the registry's supported set before it's registered.
pub fn probe_arch(path: &Path) -> Result<String> {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("safetensors"))
        .unwrap_or(false)
    {
        let cfg_path = path.parent().unwrap_or(Path::new(".")).join("config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .map_err(|e| Error::Model(format!("F16 model needs {}: {e}", cfg_path.display())))?;
        let cfg: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| Error::Model(format!("parse {}: {e}", cfg_path.display())))?;
        let mt = cfg
            .get("text_config")
            .and_then(|t| t.get("model_type"))
            .or_else(|| cfg.get("model_type"))
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        return match mt {
            "qwen3" => Ok("qwen3".into()),
            "qwen3_5" | "qwen3_5_text" => Ok("qwen35".into()),
            other => Err(Error::Model(format!(
                "safetensors model_type `{other}` has no F16 loader yet (supported: qwen3, qwen3_5)"
            ))),
        };
    }
    let mut file = File::open(path)
        .map_err(|e| Error::Model(format!("open {}: {e}", path.display())))?;
    let content = Content::read(&mut file)
        .map_err(|e| Error::Model(format!("gguf read {}: {e}", path.display())))?;
    content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .ok_or_else(|| Error::Model("GGUF missing general.architecture".into()))
}

/// Header-only probe: does this GGUF embed the RWKV **world** tokenizer
/// (`tokenizer.ggml.model == "rwkv"`)? Such GGUFs are fully self-contained —
/// a pull must SKIP the tokenizer.json download entirely (most RWKV repos
/// don't ship one; and the HF tokenizer.json ports segment text differently
/// anyway — foreign ids in, word-salad out. The embedded vocab is the only
/// authoritative one).
pub fn probe_rwkv_world(path: &Path) -> Result<bool> {
    let mut file = File::open(path)
        .map_err(|e| Error::Model(format!("open {}: {e}", path.display())))?;
    let content = Content::read(&mut file)
        .map_err(|e| Error::Model(format!("gguf read {}: {e}", path.display())))?;
    Ok(crate::tokenizer::TokenizerWrapper::is_rwkv_world(&content.metadata))
}

impl LoadedModel {
    /// Open and parse a model file — GGUF, or `.safetensors` with a sibling
    /// `config.json` (native F16/BF16 via the content shim).
    ///
    /// Reads the header + metadata table. Tensor data is left on disk and read
    /// lazily by the architecture builder (via `from_gguf`).
    pub fn load(path: &Path) -> Result<Self> {
        if path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("safetensors"))
            .unwrap_or(false)
        {
            let (arch, content, reader, meta) = crate::model::st_shim::load_safetensors(path)?;
            let name = std::fs::File::open(path.parent().unwrap_or(Path::new(".")).join("config.json"))
                .ok()
                .and_then(|_| None::<String>);
            return Ok(Self {
                arch,
                name,
                content: Some(content),
                file: ModelFile::Safetensors(reader),
                meta,
            });
        }

        let mut file = File::open(path)
            .map_err(|e| Error::Model(format!("open {}: {e}", path.display())))?;

        let content = Content::read(&mut file)
            .map_err(|e| Error::Model(format!("gguf read {}: {e}", path.display())))?;

        // Pull the architecture + a few useful fields out of the metadata table.
        let arch = content
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string())
            .ok_or_else(|| Error::Model("GGUF missing general.architecture".into()))?;

        let name = content
            .metadata
            .get("general.name")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string());

        let mut meta = HashMap::new();
        // Surface a few commonly-useful values as strings via gguf_file::Value's
        // accessor methods (each handles only its native type; try each in turn).
        for key in [
            format!("{arch}.context_length"),
            format!("{arch}.eos_token_id"),
            format!("{arch}.block_count"),
            format!("{arch}.embedding_length"),
        ] {
            if let Some(v) = content.metadata.get(&key) {
                let s: Option<String> = v
                    .to_string()
                    .ok()
                    .map(|s| s.to_string())
                    .or_else(|| v.to_u32().ok().map(|n| n.to_string()))
                    .or_else(|| v.to_i64().ok().map(|n| n.to_string()))
                    .or_else(|| v.to_f32().ok().map(|n| n.to_string()));
                if let Some(s) = s {
                    meta.insert(key, s);
                }
            }
        }

        tracing::debug!(
            "loaded GGUF: arch={arch}, name={name:?}, {} tensors, {} meta fields",
            content.tensor_infos.len(),
            meta.len(),
        );

        Ok(Self {
            arch,
            name,
            content: Some(content),
            file: ModelFile::Gguf(file),
            meta,
        })
    }

    /// Convenience: read a metadata value's string form by key.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.meta.get(key).map(|s| s.as_str())
    }

    /// Take the parsed content (consumed by the architecture builder).
    /// Returns an error if already taken.
    pub fn take_content(&mut self) -> Result<Content> {
        self.content
            .take()
            .ok_or_else(|| Error::Model("GGUF content already consumed".into()))
    }
}
