//! Model catalog — maps Ollama-style tags to local GGUF files.
//!
//! Phoenix identifies models by **verbatim tags** like `qwen2.5-coder:7b` and
//! expects those exact strings echoed back in `GET /api/tags`. The catalog is
//! the source of truth for which tags exist and where their weights live.
//!
//! Two ways a tag enters the catalog:
//! 1. **`manifest.json`** in the models dir — explicit `{ tag, file, arch }`
//!    entries. This is the authoritative form (and how a future Phoenix-driven
//!    `pull` will register files).
//! 2. **Directory scan** — every `*.gguf` file in the models dir is registered
//!    under a tag derived from its filename (e.g. `qwen2.5-coder-7b.gguf` →
//!    `qwen2.5-coder:7b`). Discovered tags can be overridden by `manifest.json`.
//!
//! The architecture is auto-detected from GGUF metadata at load time, so the
//! catalog deliberately does *not* store it (except optionally in the manifest
//! as a hint/skip-detection affordance).

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A single catalog entry: a tag → local file mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// Ollama-style tag, e.g. `qwen2.5-coder:7b`. Echoed verbatim in `/api/tags`.
    pub tag: String,
    /// Path to the GGUF file, relative to the models dir (or absolute).
    pub file: String,
    /// Optional architecture hint (e.g. `qwen2`, `llama`). If omitted, the
    /// architecture is detected from GGUF metadata at load time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arch: Option<String>,
}

/// The on-disk manifest format. Lives at `<models_dir>/manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema version; currently `1`.
    #[serde(default = "default_manifest_version")]
    pub version: u32,
    /// Explicit catalog entries. Discovered files fill in the gaps.
    #[serde(default)]
    pub models: Vec<CatalogEntry>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: default_manifest_version(),
            models: Vec::new(),
        }
    }
}

fn default_manifest_version() -> u32 {
    1
}

/// The catalog: tag → entry. Built from a manifest plus a directory scan.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    entries: BTreeMap<String, CatalogEntry>,
    /// Absolute path of the models dir entries' `file` fields are resolved against.
    models_dir: PathBuf,
}

impl Catalog {
    /// Build a catalog by reading `manifest.json` (if present) and scanning the
    /// models dir for `*.gguf` files.
    pub fn load(models_dir: &Path) -> Result<Self> {
        let mut entries: BTreeMap<String, CatalogEntry> = BTreeMap::new();

        // 1. Directory scan: register every *.gguf file under a derived tag.
        if models_dir.is_dir() {
            for entry in std::fs::read_dir(models_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                    let stem = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("model");
                    let tag = derive_tag(stem);
                    entries.insert(
                        tag.clone(),
                        CatalogEntry {
                            tag,
                            file: path
                                .file_name()
                                .and_then(|s| s.to_str())
                                .unwrap_or("model.gguf")
                                .to_string(),
                            arch: None,
                        },
                    );
                }
            }
        }

        // 2. Manifest: overrides and additional entries.
        let manifest_path = models_dir.join("manifest.json");
        if manifest_path.is_file() {
            let raw = std::fs::read_to_string(&manifest_path)?;
            let manifest: Manifest = serde_json::from_str(&raw)?;
            for entry in manifest.models {
                entries.insert(entry.tag.clone(), entry);
            }
        }

        Ok(Self {
            entries,
            models_dir: models_dir.to_path_buf(),
        })
    }

    /// All known tags, sorted (BTreeMap order).
    pub fn tags(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// All entries, sorted by tag.
    pub fn entries(&self) -> Vec<&CatalogEntry> {
        self.entries.values().collect()
    }

    /// Look up an entry by its exact tag.
    pub fn get(&self, tag: &str) -> Option<&CatalogEntry> {
        self.entries.get(tag)
    }

    /// Resolve an entry's `file` to an absolute path.
    pub fn resolve_path(&self, entry: &CatalogEntry) -> PathBuf {
        let p = Path::new(&entry.file);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.models_dir.join(&entry.file)
        }
    }

    /// Register a new entry, persisting it to `manifest.json`.
    /// Used by the `ambercore register` CLI command.
    pub fn register(&mut self, entry: CatalogEntry) -> Result<()> {
        // Ensure the models dir exists.
        std::fs::create_dir_all(&self.models_dir)?;

        // Load existing manifest (or start fresh), add/replace the entry, write back.
        let manifest_path = self.models_dir.join("manifest.json");
        let mut manifest = if manifest_path.is_file() {
            let raw = std::fs::read_to_string(&manifest_path)?;
            serde_json::from_str::<Manifest>(&raw).unwrap_or_default()
        } else {
            Manifest::default()
        };

        manifest.models.retain(|e| e.tag != entry.tag);
        manifest.models.push(entry.clone());

        let serialized = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, serialized)?;

        // Mirror into the in-memory map.
        self.entries.insert(entry.tag.clone(), entry);
        Ok(())
    }

    /// Remove a tag, returning whether it existed.
    pub fn remove(&mut self, tag: &str) -> Result<bool> {
        let existed = self.entries.remove(tag).is_some();
        if existed {
            // Persist the removal to the manifest.
            let manifest_path = self.models_dir.join("manifest.json");
            if manifest_path.is_file() {
                let raw = std::fs::read_to_string(&manifest_path)?;
                let mut manifest: Manifest =
                    serde_json::from_str(&raw).unwrap_or_default();
                manifest.models.retain(|e| e.tag != tag);
                let serialized = serde_json::to_string_pretty(&manifest)?;
                std::fs::write(&manifest_path, serialized)?;
            }
        }
        Ok(existed)
    }
}

/// Derive an Ollama-style tag from a GGUF filename stem.
///
/// Converts `qwen2.5-coder-7b` → `qwen2.5-coder:7b` by turning the last
/// `-N<size>`-shaped segment into `:tag`. Falls back to `<stem>:latest` when no
/// recognizable size suffix is present. This is a heuristic for the directory
/// scan path only — `manifest.json` entries use the tag verbatim.
fn derive_tag(stem: &str) -> String {
    // Common size suffixes: 0.5b, 1b, 1.5b, 7b, 8b, 13b, 14b, 32b, 70b, etc.
    // Also handle quantization tails like "-q4_k_m".
    let lower = stem.to_lowercase();
    for sep in lower.rsplit('-') {
        if sep.ends_with('b') && sep[..sep.len() - 1].parse::<f64>().is_ok() {
            let idx = lower.len() - sep.len() - 1; // position of the '-'
            let (head, _) = stem.split_at(idx);
            return format!("{}:{}", head, sep);
        }
    }
    format!("{}:latest", stem)
}

/// Default location of the models directory on this platform:
/// `~/.ambercore/models` (created on first use).
pub fn default_models_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| Error::NotFound("home directory".into()))?;
    Ok(home.join(".ambercore").join("models"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_tag_recognizes_size_suffix() {
        assert_eq!(derive_tag("qwen2.5-coder-7b"), "qwen2.5-coder:7b");
        assert_eq!(derive_tag("llama3-8b"), "llama3:8b");
        assert_eq!(derive_tag("phi-3.5-3.8b"), "phi-3.5:3.8b");
    }

    #[test]
    fn derive_tag_falls_back_to_latest() {
        assert_eq!(derive_tag("some-model"), "some-model:latest");
    }
}
