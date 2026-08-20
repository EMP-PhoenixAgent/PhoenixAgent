//! Configuration loading/saving and path resolution.
//!
//! Data layout under the data directory (default `~/.phoenix`):
//! ```text
//! ~/.phoenix/
//! ├── config.toml
//! ├── memory.db            <- SQLCipher-encrypted database
//! ├── salt.bin             <- Argon2 salt (needed to re-derive key)
//! └── logs/
//!     └── phoenix.log
//! ```

use std::path::PathBuf;

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
    /// its native folder (`~/.ambercore/models`).
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
    /// folder (`~/.ambercore/models`).
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
        Mode::Auto
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

    /// Default data dir: `~/.phoenix` (or `$PHOENIX_DATA_DIR` if set).
    pub fn default_data_dir() -> PathBuf {
        if let Ok(custom) = std::env::var("PHOENIX_DATA_DIR") {
            return PathBuf::from(custom);
        }
        let base = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        base.join(".phoenix")
    }

    /// Ensure the data and logs directories exist.
    pub fn ensure_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.logs_dir)?;
        Ok(())
    }
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
