//! Configuration loading/saving and path resolution.
//!
//! Phoenix is **fully portable**: every file the app creates lives inside the
//! installation folder the user picked in the installer (the directory of the
//! running executable) — no scattered home-directory folders.
//!
//! Data layout under the data directory:
//! ```text
//! <install folder>/
//! ├── config.toml
//! ├── memory.db            <- SQLCipher-encrypted database
//! ├── salt.bin             <- Argon2 salt (needed to re-derive key)
//! ├── keys.phx             <- wrapped key bundle
//! ├── 2fa_enabled          <- unencrypted boolean hint
//! ├── logs/
//! │   └── phoenix.log
//! └── models/              <- AmberCore GGUFs + tokenizers + manifest.json
//! ```
//!
//! Resolution order for the data dir ([`Paths::default_data_dir`]):
//! 1. `$PHOENIX_DATA_DIR` (tests / tooling),
//! 2. debug builds → `~/.phoenix-dev` (keeps the dev tree clean),
//! 3. release builds → the executable's folder (the install folder).
//!
//! [`migrate_legacy_data`] copies a pre-portable layout (`~/.phoenix` +
//! `~/.ambercore`) into the new data root on first run — nothing is deleted.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{PhoenixError, Result};

/// Top-level Phoenix configuration, serialized to `config.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Default model to use for new sessions.
    #[serde(default = "default_model")]
    pub model: String,

    /// Base URL of the Ollama HTTP server.
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,

    /// Base URL of the AmberCore HTTP server (the pure-Rust Ollama-compatible
    /// runner). Used when `active_backend == "ambercore"`. When
    /// [`ambercore_remote`](Self::ambercore_remote) is true this is a **remote**
    /// server URL (Phoenix does not spawn a local process).
    #[serde(default = "default_ambercore_url")]
    pub ambercore_url: String,

    /// When `true`, AmberCore runs on a **remote** server at `ambercore_url` and
    /// Phoenix must NOT spawn a local `ambercore serve` (it just points the model
    /// provider at the URL). Set by "Connect" in the AmberCore box (e.g. to link
    /// an AmberCore-Server installed on a private machine).
    #[serde(default)]
    pub ambercore_remote: bool,

    /// Which model backend is active: `"ollama"` or `"ambercore"`. Toggled from
    /// the Models panel. `resolved_provider_url()` picks the URL. Ignored when a
    /// cloud provider is active (`active_provider_id` is `Some`).
    #[serde(default = "default_active_backend")]
    pub active_backend: String,

    /// When `Some`, a cloud provider (from the `providers` table) is active and
    /// `active_backend`/`ollama_url`/`ambercore_url` are ignored. `None` = a local
    /// backend (Ollama or AmberCore) is active.
    #[serde(default)]
    pub active_provider_id: Option<i64>,

    /// Optional custom models directory for AmberCore. When unset, AmberCore uses
    /// the portable default (`<install folder>/models`).
    #[serde(default)]
    pub ambercore_models_dir: Option<String>,

    /// Optional explicit path to the `ambercore` binary. When unset, Phoenix
    /// shells out to `ambercore` on PATH.
    #[serde(default)]
    pub ambercore_binary: Option<String>,

    /// Optional explicit path to the `ollama` binary. When unset, Phoenix shells
    /// out to `ollama` on PATH.
    #[serde(default)]
    pub ollama_binary: Option<String>,

    /// Tool-approval policy for write operations (file writes, shell).
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,

    /// How many reasoning iterations the agent may take before stopping.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,

    /// Number of past messages (per session) to load into context on resume.
    #[serde(default = "default_context_window")]
    pub context_window: u32,

    /// Session-log maintenance: when true, run files that ended cleanly and
    /// contain zero error events are deleted at the next launch. Error runs
    /// and crashed runs are never touched. Default off — nothing is ever
    /// deleted silently.
    #[serde(default)]
    pub logs_auto_clean: bool,

    /// Session-log retention in days; `0` keeps files forever.
    #[serde(default)]
    pub logs_keep_days: u32,
}

impl Config {
    /// The provider URL for the currently-active backend. AmberCore's URL when
    /// `active_backend == "ambercore"`, otherwise Ollama's. Ignored when a cloud
    /// provider is active (see [`Self::is_cloud_active`]).
    pub fn resolved_provider_url(&self) -> String {
        if self.active_backend == "ambercore" {
            self.ambercore_url.clone()
        } else {
            self.ollama_url.clone()
        }
    }

    /// True when a cloud provider (rather than a local backend) is active.
    pub fn is_cloud_active(&self) -> bool {
        self.active_provider_id.is_some()
    }

    /// The resolved AmberCore models directory, or `None` to use AmberCore's native
    /// portable default (`<install folder>/models`).
    pub fn ambercore_models_dir_path(&self) -> Option<PathBuf> {
        self.ambercore_models_dir
            .as_ref()
            .filter(|s| !s.trim().is_empty())
            .map(PathBuf::from)
    }

    /// The resolved `ambercore` binary, defaulting to `"ambercore"` on PATH.
    pub fn ambercore_binary_or_default(&self) -> &str {
        self.ambercore_binary.as_deref().unwrap_or("ambercore")
    }

    /// The resolved `ollama` binary, defaulting to `"ollama"` on PATH.
    pub fn ollama_binary_or_default(&self) -> &str {
        self.ollama_binary.as_deref().unwrap_or("ollama")
    }
}

fn default_model() -> String {
    "qwen2.5-coder:7b".to_string()
}
fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}
fn default_ambercore_url() -> String {
    "http://localhost:42069".to_string()
}
fn default_active_backend() -> String {
    "ollama".to_string()
}
fn default_max_iterations() -> u32 {
    25
}
fn default_context_window() -> u32 {
    50
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: default_model(),
            ollama_url: default_ollama_url(),
            ambercore_url: default_ambercore_url(),
            ambercore_remote: false,
            active_backend: default_active_backend(),
            active_provider_id: None,
            ambercore_models_dir: None,
            ambercore_binary: None,
            ollama_binary: None,
            approval_policy: ApprovalPolicy::default(),
            max_iterations: default_max_iterations(),
            context_window: default_context_window(),
            logs_auto_clean: false,
            logs_keep_days: 0,
        }
    }
}

/// Controls when the agent must ask the user for approval before a tool runs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalPolicy {
    /// Ask before any tool call, including reads.
    All,
    /// Ask before anything that mutates state (file writes, shell). Reads run freely.
    WritesOnly,
    /// Never ask — run everything automatically (dangerous).
    Never,
}

impl Default for ApprovalPolicy {
    fn default() -> Self {
        ApprovalPolicy::WritesOnly
    }
}

impl ApprovalPolicy {
    pub fn requires_approval(&self, kind: ToolKind) -> bool {
        match self {
            ApprovalPolicy::All => true,
            ApprovalPolicy::WritesOnly => matches!(kind, ToolKind::Write),
            ApprovalPolicy::Never => false,
        }
    }
}

/// The reasoning mode the agent is operating in — a UI-facing preset (selected
/// from the chat mode bar above the send button) that maps to an
/// [`ApprovalPolicy`], the live-switchable behavior lever.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Read-only planning: investigate files + context, propose a plan, don't edit.
    Plan,
    /// Discussion: talk through the task with the user, don't execute tools.
    Think,
    /// Autonomous: full access, act directly.
    Auto,
}

impl Default for Mode {
    fn default() -> Self {
        Mode::Think
    }
}

impl Mode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "plan" => Some(Mode::Plan),
            "think" => Some(Mode::Think),
            "auto" => Some(Mode::Auto),
            _ => None,
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            Mode::Plan => "Plan",
            Mode::Think => "Think",
            Mode::Auto => "Auto",
        }
    }
    /// The approval policy this mode runs under.
    pub fn approval_policy(&self) -> ApprovalPolicy {
        match self {
            Mode::Plan => ApprovalPolicy::WritesOnly,
            Mode::Think => ApprovalPolicy::All,
            Mode::Auto => ApprovalPolicy::Never,
        }
    }

    /// The behavior directive injected into the system prompt for this mode (so
    /// the model self-limits, on top of the approval-policy enforcement).
    pub fn directive(&self) -> &'static str {
        match self {
            Mode::Plan => "READ-ONLY planning. Investigate the codebase and context with \
                           read/search tools; do NOT modify, write, or run state-changing \
                           commands. Produce a plan and explain it.",
            Mode::Think => "DISCUSSION. Talk through the task with the user to clarify goals \
                            and trade-offs. Do not execute tools or make changes — converse \
                            and advise.",
            Mode::Auto => "AUTONOMOUS. You have full access — act directly to complete the \
                           task. Only pause to ask the user if you are genuinely stuck or need \
                           a decision.",
        }
    }
}

/// Coarse classification of a tool's effect, used by the approval gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolKind {
    /// Read-only — no state mutation.
    Read,
    /// Mutates state (writes files, runs shell, etc.).
    Write,
}

/// Resolved runtime paths and loaded configuration.
#[derive(Debug, Clone)]
pub struct Paths {
    /// Root data directory (`~/.phoenix` by default).
    pub data_dir: PathBuf,
    pub config_path: PathBuf,
    pub db_path: PathBuf,
    pub salt_path: PathBuf,
    pub logs_dir: PathBuf,
    /// Unencrypted marker file: just `"1"` or `"0"`. Tells the unlock screen
    /// whether to show the 2FA code field WITHOUT needing to open the encrypted
    /// DB. Contains NO secret — only a boolean hint.
    pub totp_flag_path: PathBuf,
    /// Unencrypted key bundle: holds the SQLCipher DB key **wrapped** (AES-256-GCM)
    /// under the user's launch password, plus an optional recovery wrap under the
    /// 2FA seed. Contains NO plaintext key — only ciphertext + salts. Readable
    /// pre-unlock so the launch gate can run before the DB is open.
    pub key_bundle_path: PathBuf,
}

impl Paths {
    /// Resolve paths for the given data directory.
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            config_path: data_dir.join("config.toml"),
            db_path: data_dir.join("memory.db"),
            salt_path: data_dir.join("salt.bin"),
            logs_dir: data_dir.join("logs"),
            totp_flag_path: data_dir.join("2fa_enabled"),
            key_bundle_path: data_dir.join("keys.phx"),
            data_dir,
        }
    }

    /// Default data dir — see the module docs for the resolution order:
    /// `$PHOENIX_DATA_DIR`, then `~/.phoenix-dev` in debug builds, then —
    /// this Encapsulated build — the **staging dir**: a private temp folder
    /// keyed by the exe's own path (`%TEMP%\PA-Encaps\<hash>`). All state
    /// lives there at runtime and is sealed back INTO the exe at exit /
    /// auto-save, so the app folder holds nothing but the exe itself.
    /// (`PHOENIX_ENCAPS_PORTABLE=1` restores the classic exe-dir layout.)
    pub fn default_data_dir() -> PathBuf {
        if let Ok(custom) = std::env::var("PHOENIX_DATA_DIR") {
            return PathBuf::from(custom);
        }
        #[cfg(debug_assertions)]
        {
            let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
            base.join(".phoenix-dev")
        }
        #[cfg(not(debug_assertions))]
        {
            if std::env::var_os("PHOENIX_ENCAPS_PORTABLE").is_some() {
                return std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                    .unwrap_or_else(|| PathBuf::from("."));
            }
            Self::encaps_staging_dir()
        }
    }

    /// The Encapsulated staging dir: stable per exe LOCATION (copying the exe
    /// to another folder gives a fresh, isolated staging), tucked into the
    /// user's temp where it never pollutes the app folder.
    pub fn encaps_staging_dir() -> PathBuf {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("pa-encaps"));
        let mut h = DefaultHasher::new();
        exe.to_string_lossy().to_lowercase().hash(&mut h);
        let key = format!("{:016x}", h.finish());
        std::env::temp_dir().join("PA-Encaps").join(key)
    }

    /// Ensure the data and logs directories exist **and are writable** — the
    /// portable layout keeps everything in the install folder, so a folder the
    /// user can't write to (e.g. a system location) must fail loudly here, not
    /// as a mysterious DB error later.
    pub fn ensure_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(&self.data_dir).map_err(|e| self.unwritable(e))?;
        std::fs::create_dir_all(&self.logs_dir).map_err(|e| self.unwritable(e))?;
        let probe = self.data_dir.join(".write-probe");
        std::fs::write(&probe, b"ok")
            .and_then(|_| std::fs::remove_file(&probe))
            .map_err(|e| self.unwritable(e))?;
        Ok(())
    }

    fn unwritable(&self, e: std::io::Error) -> PhoenixError {
        PhoenixError::Config(format!(
            "the data folder {} is not writable ({e}). Phoenix stores all of its \
             files in the installation folder — reinstall it to a location your \
             user account can write to.",
            self.data_dir.display(),
        ))
    }
}

/// The default AmberCore models dir: `<data root>/models` — inside the
/// installation folder like everything else (a custom dir set in the Models
/// panel still wins).
pub fn default_models_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("models")
}

/// The Encapsulated default models dir: a plain `models` folder **beside the
/// exe** — public artifacts that live and stay on disk, never inside the
/// encrypted capsule and never in the temp staging area. Created on
/// resolution. Falls back to the data-dir default when the exe folder is
/// unwritable (or in debug builds, where the "exe" is `target/debug` and
/// models belong in the dev data dir instead).
pub fn encaps_models_dir(data_dir: &Path) -> PathBuf {
    #[cfg(not(debug_assertions))]
    {
        if std::env::var_os("PHOENIX_ENCAPS_PORTABLE").is_none() {
            if let Some(dir) = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join("models")))
            {
                if dir_writable(&dir) {
                    return dir;
                }
            }
        }
    }
    let dir = default_models_dir(data_dir);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Ensure `dir` exists and is writable (probe write). Used by
/// [`encaps_models_dir`] and [`crate::logsys::encaps_logs_dir`] — a folder the
/// app cannot write to must fall back, not fail every write later.
pub(crate) fn dir_writable(dir: &Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(".write-probe");
    std::fs::write(&probe, b"ok")
        .and_then(|_| std::fs::remove_file(&probe))
        .is_ok()
}

/// One-time move of models from the old staging layout (`<temp staging>/
/// models`, where the pre-Phase-C build kept them) into the exe-side models
/// folder. Best-effort: per-item failures are logged and skipped; `rename`
/// first (fast, same volume), then a streamed copy across volumes. Never
/// overwrites, never deletes a source that failed to copy.
pub fn migrate_staging_models(staging_models: &Path, dest: &Path) {
    let _ = std::fs::create_dir_all(dest);
    // Same dir (read-only-exe fallback resolves both to `<data dir>/models`)
    // — nothing to move, and the per-item "already there" branch must NOT
    // run against itself.
    let (Ok(a), Ok(b)) = (staging_models.canonicalize(), dest.canonicalize()) else {
        return;
    };
    if a == b {
        return;
    }
    let Ok(rd) = std::fs::read_dir(staging_models) else {
        return;
    };
    let _ = std::fs::create_dir_all(dest);
    for entry in rd.flatten() {
        let name = entry.file_name();
        let from = entry.path();
        let to = dest.join(&name);
        if to.exists() {
            let _ = std::fs::remove_dir_all(&from); // already there — drop the dup
            continue;
        }
        let moved = std::fs::rename(&from, &to).is_ok() || copy_dir_streamed(&from, &to);
        if moved {
            tracing::info!("models: moved {} beside the exe", name.to_string_lossy());
        } else {
            tracing::warn!(
                "models: could not move {} — it stays in the temp staging area",
                name.to_string_lossy()
            );
        }
    }
    let _ = std::fs::remove_dir(staging_models); // no-op unless now empty
}

/// Recursive copy used by [`migrate_staging_models`] (streamed, never
/// buffered in full). Returns success only when every file copied.
fn copy_dir_streamed(src: &Path, dst: &Path) -> bool {
    if src.is_file() {
        if let Some(p) = dst.parent() {
            if std::fs::create_dir_all(p).is_err() {
                return false;
            }
        }
        let Ok(mut s) = std::fs::File::open(src) else { return false };
        let Ok(mut d) = std::fs::File::create(dst) else { return false };
        if std::io::copy(&mut s, &mut d).is_err() {
            let _ = std::fs::remove_file(dst);
            return false;
        }
        return std::fs::remove_file(src).is_ok();
    }
    let Ok(rd) = std::fs::read_dir(src) else { return false };
    if std::fs::create_dir_all(dst).is_err() {
        return false;
    }
    rd.flatten().all(|e| copy_dir_streamed(&e.path(), &dst.join(e.file_name())))
}

/// One-time migration from the pre-portable layout into `paths.data_dir`.
///
/// Runs when the new data root has no `config.toml` yet. Copies (never
/// deletes) from `legacy_home/.phoenix` (config, DB, salt, keys, 2FA marker)
/// and `legacy_home/.ambercore/models` (GGUFs, tokenizers, manifest) into the
/// new root; a migrated config pointing at the old default models dir is
/// rewritten to use the new one. Existing files in the destination are never
/// overwritten.
pub fn migrate_legacy_data(paths: &Paths, legacy_home: &Path) {
    if paths.config_path.exists() {
        return; // Fresh install or already migrated.
    }
    let legacy_phoenix = legacy_home.join(".phoenix");
    let legacy_models = legacy_home.join(".ambercore").join("models");
    if !legacy_phoenix.is_dir() && !legacy_models.is_dir() {
        return; // Nothing to migrate.
    }
    tracing::info!(
        new_root = %paths.data_dir.display(),
        "first run after the portable-layout change: migrating legacy data"
    );

    if legacy_phoenix.is_dir() {
        for name in ["config.toml", "memory.db", "salt.bin", "keys.phx", "2fa_enabled"] {
            let src = legacy_phoenix.join(name);
            let dst = paths.data_dir.join(name);
            if src.is_file() && !dst.exists() {
                if let Err(e) = std::fs::copy(&src, &dst) {
                    tracing::warn!("migrate {name}: {e}");
                } else {
                    tracing::info!("migrated {name}");
                }
            }
        }
    }

    let new_models = default_models_dir(&paths.data_dir);
    if legacy_models.is_dir() {
        if let Err(e) = copy_dir_missing(&legacy_models, &new_models) {
            tracing::warn!("migrate models: {e}");
        }
    }

    // A migrated config may still point at the old default models dir —
    // rewrite it to the portable one (a truly custom dir is left alone).
    if let Ok(mut cfg) = load_config(paths) {
        let rewrite = cfg
            .ambercore_models_dir
            .as_deref()
            .map(|d| PathBuf::from(d) == legacy_models)
            .unwrap_or(false);
        if rewrite {
            cfg.ambercore_models_dir = None;
            let _ = save_config(paths, &cfg);
            tracing::info!("migrated config: models dir reset to the portable default");
        }
    }
}

/// Recursively copy `src` into `dst`, skipping files that already exist at the
/// destination (never overwrites). Creates `dst` as needed.
fn copy_dir_missing(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let s = entry.path();
        let d = dst.join(entry.file_name());
        if s.is_dir() {
            copy_dir_missing(&s, &d)?;
        } else if !d.exists() {
            std::fs::copy(&s, &d)?;
        }
    }
    Ok(())
}

/// Load config from disk, falling back to defaults if missing.
pub fn load_config(paths: &Paths) -> Result<Config> {
    if !paths.config_path.exists() {
        return Ok(Config::default());
    }
    let text = std::fs::read_to_string(&paths.config_path)?;
    let cfg: Config = toml::from_str(&text)
        .map_err(|e| PhoenixError::Config(format!("parse {}: {e}", paths.config_path.display())))?;
    Ok(cfg)
}

/// Persist config to disk.
pub fn save_config(paths: &Paths, cfg: &Config) -> Result<()> {
    let text = toml::to_string_pretty(cfg)
        .map_err(|e| PhoenixError::Config(format!("serialize config: {e}")))?;
    std::fs::write(&paths.config_path, text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_staging_models_moves_children() {
        let tmp = std::env::temp_dir().join(format!("pa-mig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let src = tmp.join("staging/models/qwen-3");
        let dst = tmp.join("exedir/models");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("model.gguf"), b"gguf").unwrap();
        std::fs::write(src.join("model.tokenizer.json"), b"tk").unwrap();

        migrate_staging_models(&tmp.join("staging/models"), &dst);
        assert!(dst.join("qwen-3/model.gguf").exists());
        assert!(dst.join("qwen-3/model.tokenizer.json").exists());
        assert!(!src.exists(), "the moved tree is gone from staging");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn migrate_staging_models_same_dir_is_a_noop() {
        let tmp = std::env::temp_dir().join(format!("pa-mig-same-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let dir = tmp.join("models/qwen-3");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.gguf"), b"gguf").unwrap();

        // src == dest (the read-only-exe fallback): must not delete anything.
        migrate_staging_models(&tmp.join("models"), &tmp.join("models"));
        assert!(dir.join("model.gguf").exists(), "same-dir migrate must be a no-op");
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
