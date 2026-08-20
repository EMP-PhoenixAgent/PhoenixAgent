//! Health monitoring: background probes for every component Phoenix depends on.
//!
//! A background task pings the Ollama server, the configured model, the
//! encrypted database, and the shell/ripgrep tools every few seconds and
//! publishes the aggregate [`HealthState`] through a channel. The TUI renders
//! the latest snapshot in a bottom toolbar (green = healthy, red = down) so
//! you can see at a glance that every connection is alive.

use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use tokio::sync::{mpsc, watch, Mutex};
use tokio::time;

use crate::db::MemoryStore;
use crate::model::ModelProvider;

/// Health of a single component.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", content = "detail", rename_all = "snake_case")]
pub enum ComponentStatus {
    /// Not yet probed.
    Unknown,
    /// Probe in progress.
    Checking,
    /// Reachable / functional. Carries a short detail string.
    Ok(String),
    /// Unreachable / broken. Carries the reason.
    Down(String),
}

impl Default for ComponentStatus {
    fn default() -> Self {
        ComponentStatus::Unknown
    }
}

impl ComponentStatus {
    pub fn healthy(&self) -> bool {
        matches!(self, ComponentStatus::Ok(_))
    }
}

/// A snapshot of every monitored component.
#[derive(Debug, Clone, Default, Serialize)]
pub struct HealthState {
    pub ollama: ComponentStatus,
    pub model: ComponentStatus,
    pub database: ComponentStatus,
    pub ripgrep: ComponentStatus,
    pub shell: ComponentStatus,
}

impl HealthState {
    /// Number of components currently healthy.
    pub fn healthy_count(&self) -> usize {
        [
            self.ollama.healthy(),
            self.model.healthy(),
            self.database.healthy(),
            self.ripgrep.healthy(),
            self.shell.healthy(),
        ]
        .iter()
        .filter(|&&h| h)
        .count()
    }

    /// Total number of monitored components.
    pub fn total(&self) -> usize {
        5
    }

    /// The components in display order.
    pub fn components(&self) -> [(&'static str, &ComponentStatus); 5] {
        [
            ("Ollama", &self.ollama),
            ("Model", &self.model),
            ("DB", &self.database),
            ("ripgrep", &self.ripgrep),
            ("shell", &self.shell),
        ]
    }
}

/// Spawn the background health monitor. Returns a receiver yielding the latest
/// [`HealthState`] after each probe cycle (every `interval_secs` seconds).
///
/// `model_rx` is a watch channel that carries the *currently active* model
/// name, so the "model pulled?" probe reflects live model switches rather than
/// the model that was active at boot.
///
/// The monitor stops automatically when the receiver is dropped (i.e. when the
/// TUI exits).
pub fn spawn_monitor(
    provider: Arc<dyn ModelProvider>,
    model_rx: watch::Receiver<String>,
    store: Arc<Mutex<MemoryStore>>,
    interval_secs: u64,
) -> mpsc::Receiver<HealthState> {
    let (tx, rx) = mpsc::channel::<HealthState>(4);
    tokio::spawn(async move {
        // Send an initial "checking" snapshot immediately so the toolbar isn't
        // blank during the first probe.
        let checking = HealthState {
            ollama: ComponentStatus::Checking,
            model: ComponentStatus::Checking,
            database: ComponentStatus::Checking,
            ripgrep: ComponentStatus::Checking,
            shell: ComponentStatus::Checking,
        };
        let _ = tx.send(checking).await;

        let mut ticker = time::interval(Duration::from_secs(interval_secs));
        // The first tick() completes immediately; skip it so we don't probe
        // twice in quick succession after the initial "checking" send.
        ticker.tick().await;

        loop {
            ticker.tick().await;
            // Read the latest model name without blocking.
            let model = model_rx.borrow().clone();
            let snapshot = probe_all(&provider, &model, &store).await;
            if tx.send(snapshot).await.is_err() {
                break; // TUI gone; stop the monitor.
            }
        }
    });
    rx
}

/// Run one full probe cycle across every component.
async fn probe_all(
    provider: &Arc<dyn ModelProvider>,
    model: &str,
    store: &Arc<Mutex<MemoryStore>>,
) -> HealthState {
    // Ollama/AmberCore reachability + model availability, derived from a single call.
    let (ollama, model_status) = match provider.list_models().await {
        Ok(models) => {
            let ollama = ComponentStatus::Ok(format!("{} model(s)", models.len()));
            let has = models.iter().any(|m| m == model);
            let model_status = if has {
                // If the backend exposes throughput stats (AmberCore), append
                // tokens/sec to the model detail so it shows in the health bar.
                let mut detail = model.to_string();
                if let Some(stats) = provider.stats().await {
                    if let Some(tps) = stats.tokens_per_sec {
                        detail.push_str(&format!(" · {tps:.1} T/s"));
                    }
                }
                ComponentStatus::Ok(detail)
            } else {
                ComponentStatus::Down(format!("'{model}' not pulled"))
            };
            (ollama, model_status)
        }
        Err(e) => (
            ComponentStatus::Down(short_reason(&e.to_string())),
            ComponentStatus::Down("server unreachable".into()),
        )
    };

    // Encrypted DB: a trivial round-trip confirms it's unlocked and responsive.
    let database = {
        let store = store.lock().await;
        match store.conn().query_row("SELECT 1", [], |row| row.get::<_, i64>(0)) {
            Ok(_) => ComponentStatus::Ok("unlocked".into()),
            Err(e) => ComponentStatus::Down(short_reason(&e.to_string())),
        }
    };

    // ripgrep presence (used by the grep tool).
    let ripgrep = probe_command("rg", &["--version"]).await;

    // Shell smoke test (used by run_command).
    let (program, flag) = if cfg!(target_os = "windows") {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    };
    let shell = probe_command(program, &[flag, "echo ok"]).await;

    HealthState {
        ollama,
        model: model_status,
        database,
        ripgrep,
        shell,
    }
}

/// Run a command with a short timeout; return Ok if it exits successfully.
async fn probe_command(program: &str, args: &[&str]) -> ComponentStatus {
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args);
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .stdin(std::process::Stdio::null());
    match time::timeout(Duration::from_secs(3), cmd.status()).await {
        Ok(Ok(status)) if status.success() => ComponentStatus::Ok("present".into()),
        Ok(Ok(_)) => ComponentStatus::Down("non-zero exit".into()),
        Ok(Err(_)) => ComponentStatus::Down("not found".into()),
        Err(_) => ComponentStatus::Down("timeout".into()),
    }
}

/// Trim an error message to a short single-line reason (UTF-8 safe).
fn short_reason(s: &str) -> String {
    let first_line = s.lines().next().unwrap_or(s);
    let chars: Vec<char> = first_line.chars().collect();
    if chars.len() > 32 {
        let mut t: String = chars[..32].iter().collect();
        t.push('…');
        t
    } else {
        first_line.to_string()
    }
}
