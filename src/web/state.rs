//! Managed application state for the Tauri web layer.
//!
//! `WebState` holds everything the Tauri commands need access to. Most fields
//! are `Option` and start empty — they're populated by the `unlock` command
//! once the user supplies a passphrase and the backend is wired up.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{mpsc, watch, Mutex};

use super::SharedStore;
use crate::agent::Command;
use crate::backend::process::ProcessManager;
use crate::config::{Config, Paths};
use crate::crypto::DerivedKey;
use crate::health::HealthState;
use crate::model::dispatch::{ActiveRoute, DispatchProvider, LocalBackend};
use crate::model::ModelProvider;

pub struct WebState {
    /// Channel to send commands to the agent runtime. `None` until unlock.
    pub cmd_tx: Mutex<Option<mpsc::Sender<Command>>>,
    /// The encrypted store handle. `None` until unlock.
    pub store: Mutex<Option<SharedStore>>,
    /// The dispatching provider — routes `chat()`/`list_models()` to the active
    /// backend (local Ollama/AmberCore or a cloud provider). Stored concretely so
    /// commands can call `set_local`/`set_cloud`; cloned into the runtime +
    /// health monitor as `Arc<dyn ModelProvider>`. Available from launch.
    pub provider: Arc<DispatchProvider>,
    /// Manages the active backend server process (AmberCore/Ollama `serve`).
    pub process_mgr: ProcessManager,
    /// Latest health snapshot, updated by the forwarder task.
    pub health: Mutex<HealthState>,
    /// Loaded config.
    pub config: Mutex<Config>,
    /// Resolved filesystem paths.
    pub paths: Paths,
    /// Working directory / project root. Mutable because the sidebar workdir
    /// selector can change it at runtime.
    pub workdir: Mutex<PathBuf>,
    /// Watch channel carrying the currently active model name, so both the
    /// agent runtime and the health monitor observe live model switches.
    /// `None` until `boot_runtime` creates it.
    pub model_tx: Mutex<Option<watch::Sender<String>>>,
    /// The in-memory DB decryption key, retained while unlocked so the 2FA
    /// recovery wrap and DB-rekey re-wrap can be maintained without re-asking
    /// for the launch password. `None` until unlock. Zeroized on drop.
    pub db_key: Mutex<Option<DerivedKey>>,
    /// Current reasoning mode (Plan/Think/Auto) — drives the approval-policy
    /// preset selected from the chat mode bar. Readable from launch (default
    /// Think); the runtime applies it live via `Command::SetMode`.
    pub mode: Mutex<crate::config::Mode>,
}

impl WebState {
    pub fn new(config: Config, paths: Paths, workdir: PathBuf) -> Self {
        // Resolve the starting route + local URL from config. If a cloud provider
        // is active we still seed the local provider's URL (harmless; overwritten
        // on the first cloud chat) but the route points at cloud. The concrete
        // provider + key for a cloud route are pushed in by `boot_runtime` once
        // the DB is open (we can't read the key until unlock).
        let local_url = config.resolved_provider_url();
        let local_backend = LocalBackend::parse(&config.active_backend)
            .unwrap_or(LocalBackend::Ollama);
        let route = if config.is_cloud_active() {
            // Cloud active at startup — local backend will be shadowed. The exact
            // cloud endpoint is re-applied after unlock in `boot_runtime`.
            ActiveRoute::default()
        } else {
            ActiveRoute::Local { backend: local_backend }
        };
        // The embedded AmberCore engine is constructed once at launch (cheap —
        // models load lazily on first use / warm-up). Models default to
        // `<install folder>/models` (the portable layout); a custom dir set in
        // the Models panel wins. A missing/unreadable dir degrades to a temp
        // catalog so the app still boots.
        let models_dir = config
            .ambercore_models_dir_path()
            .unwrap_or_else(|| crate::config::default_models_dir(&paths.data_dir));
        let embedded = crate::model::ambercore_embedded::EmbeddedAmberCore::new(Some(models_dir))
            .unwrap_or_else(|e| {
                tracing::warn!("embedded AmberCore init failed; falling back to a temp catalog: {e}");
                let dir = std::env::temp_dir().join("phoenix-ambercore-fallback");
                crate::model::ambercore_embedded::EmbeddedAmberCore::new(Some(dir))
                    .expect("embedded AmberCore fallback engine must construct")
            });
        let provider = Arc::new(DispatchProvider::new(&local_url, route, embedded));

        Self {
            cmd_tx: Mutex::new(None),
            store: Mutex::new(None),
            provider,
            process_mgr: ProcessManager::new(),
            health: Mutex::new(HealthState::default()),
            config: Mutex::new(config),
            paths,
            workdir: Mutex::new(workdir),
            model_tx: Mutex::new(None),
            db_key: Mutex::new(None),
            mode: Mutex::new(crate::config::Mode::default()),
        }
    }

    /// View the provider as a trait object (for handing clones to the runtime +
    /// health monitor).
    pub fn provider_dyn(&self) -> Arc<dyn ModelProvider> {
        self.provider.clone()
    }
}
