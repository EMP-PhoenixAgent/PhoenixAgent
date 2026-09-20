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
    /// models dir for `*.gguf` files — flat files **and** one level of
    /// per-model subfolders (the Phoenix pull layout: `<model-name>/<model>.gguf`).
    pub fn load(models_dir: &Path) -> Result<Self> {
        let mut entries: BTreeMap<String, CatalogEntry> = BTreeMap::new();

        // 1. Directory scan: register every *.gguf file under a derived tag.
        if models_dir.is_dir() {
            for entry in std::fs::read_dir(models_dir)? {
                let entry = entry?;
                let path = entry.path();
                if is_model_file(&path) && is_first_shard(&path) {
                    insert_scanned(&mut entries, &path, None);
                } else if path.is_dir() {
                    // One level of per-model subfolders. Relative `file` paths
                    // (`<folder>/<name>.gguf`) resolve against the models dir.
                    for sub in std::fs::read_dir(&path)? {
                        let sub = sub?.path();
                        if is_model_file(&sub) && is_first_shard(&sub) {
                            let folder = path
                                .file_name()
                                .and_then(|n| n.to_str())
                                .unwrap_or_default();
                            insert_scanned(&mut entries, &sub, Some(folder));
                        }
                    }
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

    /// Look up an entry tolerating the spellings the ecosystem produces:
    /// Phoenix registers pulled models under the bare file stem
    /// (`gemma-4-E2B-it-Q4_K_M`) while the directory scan derives
    /// size-suffixed tags (`Qwen3.5:4b`) or `<stem>:latest`. Resolution tries
    /// the exact tag, then the stem before the first `:`, then `tag:latest`
    /// (when the tag has no `:`), and finally a case-insensitive match on the
    /// underlying FILE STEM — stable across pull-registered and scan-derived
    /// spellings, so an emptied manifest (or a renamed file) still resolves
    /// the model the config asks for. Returns the entry and the canonical
    /// registered tag — used to key replica pools so alias spellings share
    /// one pool.
    pub fn resolve(&self, tag: &str) -> Option<(&CatalogEntry, String)> {
        if let Some(entry) = self.entries.get(tag) {
            return Some((entry, tag.to_string()));
        }
        if let Some((stem, _)) = tag.split_once(':') {
            if let Some(entry) = self.entries.get(stem) {
                return Some((entry, stem.to_string()));
            }
        } else if let Some(entry) = self.entries.get(&format!("{tag}:latest")) {
            return Some((entry, format!("{tag}:latest")));
        }
        let wanted = normalize_stem(tag);
        if !wanted.is_empty() {
            for (key, entry) in &self.entries {
                if normalize_stem(&entry.file) == wanted {
                    return Some((entry, key.clone()));
                }
            }
        }
        None
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

/// Whether a path has a `.gguf` extension (case-insensitive).
/// A model file the catalog registers: quantized GGUF, or native F16
/// safetensors (needs a sibling `config.json` to load — enforced at load
/// time, not here).
fn is_model_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("gguf") || e.eq_ignore_ascii_case("safetensors"))
        .unwrap_or(false)
}

/// Whether a scanned model file should be REGISTERED: sharded safetensors
/// checkpoints contribute only their first shard — the engine stitches the
/// rest via the sibling index.json, and the catalog stays one-entry-per-model.
fn is_first_shard(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return true };
    let lower = name.to_ascii_lowercase();
    if !lower.ends_with(".safetensors") {
        return true;
    }
    match lower.rfind("-of-") {
        None => true,
        Some(_) => lower.contains("-00001-of-"),
    }
}

/// Register a scanned GGUF under its derived tag. `folder` prefixes the stored
/// (relative) `file` path when the model lives in a per-model subfolder.
fn insert_scanned(
    entries: &mut BTreeMap<String, CatalogEntry>,
    path: &Path,
    folder: Option<&str>,
) {
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.gguf");
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    let tag = derive_tag(stem);
    let file = match folder {
        Some(f) => format!("{f}/{filename}"),
        None => filename.to_string(),
    };
    entries.insert(tag.clone(), CatalogEntry { tag, file, arch: None });
}

/// Lowercased GGUF file stem shared by a request name and a catalog entry's
/// `file` path — the common denominator between pull-registered names
/// (`Qwen3.5-4B-Q4_K_M`) and scan-derived tags (`Qwen3.5:4b`, whose `file` is
/// `Qwen3.5-4B-Q4_K_M/Qwen3.5-4B-Q4_K_M.gguf`). Used by [`Catalog::resolve`]'s
/// final fallback.
fn normalize_stem(name_or_tag: &str) -> String {
    // Strip a `:tag` suffix from the REQUEST side; the `file` side is a path.
    let base = name_or_tag.split(':').next().unwrap_or(name_or_tag);
    let file_name = base.rsplit(['/', '\\']).next().unwrap_or(base);
    let stem = file_name.strip_suffix(".gguf").unwrap_or(file_name);
    stem.trim().to_lowercase()
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

    #[test]
    fn resolve_tolerates_both_tag_spellings() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Scan-registered: tag is `<stem>:latest`, but Phoenix references the
        // bare stem (its `derive_ambercore_tag` keeps the stem verbatim).
        std::fs::write(dir.path().join("gemma-4-E2B-it-Q4_K_M.gguf"), b"x").unwrap();
        let mut cat = Catalog::load(dir.path()).unwrap();
        assert!(cat.get("gemma-4-E2B-it-Q4_K_M").is_none(), "bare stem is not the exact tag");
        let (_, canonical) = cat
            .resolve("gemma-4-E2B-it-Q4_K_M")
            .expect("bare stem resolves to the :latest entry");
        assert_eq!(canonical, "gemma-4-E2B-it-Q4_K_M:latest");

        // Manifest-registered: Phoenix writes the bare stem; a `:latest`
        // client spelling still finds it and reports the canonical tag.
        cat.register(CatalogEntry {
            tag: "kimi-k2-instruct-Q4_K_M".into(),
            file: "kimi/kimi-k2-instruct-Q4_K_M.gguf".into(),
            arch: None,
        })
        .unwrap();
        let (_, canonical) = cat
            .resolve("kimi-k2-instruct-Q4_K_M:latest")
            .expect(":latest spelling resolves to the bare-stem entry");
        assert_eq!(canonical, "kimi-k2-instruct-Q4_K_M");
    }

    #[test]
    fn scan_finds_flat_and_subfolder_models() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Flat model (the pre-subfolder layout — still supported).
        std::fs::write(dir.path().join("flat-model-7b.gguf"), b"x").unwrap();
        // Per-model subfolder (the Phoenix pull layout).
        let sub = dir.path().join("gemma3-it");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("gemma3-it-q4.gguf"), b"x").unwrap();
        // Non-GGUF files (tokenizers, manifests) must be ignored by the scan.
        std::fs::write(sub.join("gemma3-it-q4.tokenizer.json"), b"{}").unwrap();

        let cat = Catalog::load(dir.path()).unwrap();
        assert!(cat.get("flat-model:7b").is_some(), "flat model registered");
        let entry = cat.get("gemma3-it-q4:latest").expect("subfolder model registered");
        assert_eq!(entry.file, "gemma3-it/gemma3-it-q4.gguf");
        // Resolves against the models dir with the subfolder in the path.
        let resolved = cat.resolve_path(entry);
        assert_eq!(resolved, sub.join("gemma3-it-q4.gguf"));
    }
}
