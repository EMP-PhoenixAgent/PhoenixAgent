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
use std::path::Path;

/// A loaded GGUF model, ready to be handed to an architecture constructor.
///
/// Owns the open file (so tensors can be read lazily from disk by the
/// architecture builder) plus the parsed [`Content`] (consumed by `from_gguf`).
pub struct LoadedModel {
    /// Architecture string from `general.architecture` (e.g. `"qwen2"`, `"llama"`).
    pub arch: String,
    /// Human-readable name from `general.name` (best-effort).
    pub name: Option<String>,
    /// The parsed GGUF content — handed to `ModelWeights::from_gguf`. It is an
    /// `Option` because the architecture builder consumes it.
    pub content: Option<Content>,
    /// The still-open file handle, kept so the builder can read tensor data.
    pub file: File,
    /// Selected metadata values surfaced for convenience (max seq len, etc.).
    pub meta: HashMap<String, String>,
}

/// Read only a GGUF's header and return its `general.architecture` string.
///
/// Cheap regardless of model size (the header + metadata table is a few MB;
/// tensor data is never read). Used to validate a freshly downloaded model
/// against the registry's supported set before it's registered.
pub fn probe_arch(path: &Path) -> Result<String> {
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

impl LoadedModel {
    /// Open and parse a GGUF file.
    ///
    /// Reads the header + metadata table. Tensor data is left on disk and read
    /// lazily by the architecture builder (via `from_gguf`).
    pub fn load(path: &Path) -> Result<Self> {
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
            file,
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
