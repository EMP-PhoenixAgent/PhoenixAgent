//! Manages the active backend server process.
//!
//! A single [`ProcessManager`] tracks at most one long-running server process —
//! **Ollama**. (Local AmberCore no longer needs one: it runs *in-process*,
//! compiled into the Phoenix binary; remote AmberCore is pure HTTP.) Starting a
//! server stops any running one first, honoring the Models-panel contract that
//! "every Run button closes other's process for efficiency purposes".
//!
//! Spawn follows the same `tokio::process::Command` idiom as the shell tool, but
//! the child is **detached** (we keep its handle rather than awaiting its output)
//! and its stdout/stderr are piped to a background line-reader that forwards
//! each line to `tracing` (so server logs land in the Phoenix log file).

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::error::{PhoenixError, Result};

/// Which server the manager currently has running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunningBackend {
    Ollama,
}

impl RunningBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            RunningBackend::Ollama => "ollama",
        }
    }
}

impl fmt::Display for RunningBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A tracked server child + what it is.
struct ServerHandle {
    child: Child,
    backend: RunningBackend,
}

/// Owns the single active backend server process (if any).
///
/// Shared across the web layer via `Arc` (stored in [`crate::web::state::WebState`]).
#[derive(Clone)]
pub struct ProcessManager {
    inner: Arc<Mutex<Option<ServerHandle>>>,
}

impl Default for ProcessManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(None)),
        }
    }

    /// What server (if any) is currently running.
    pub async fn running(&self) -> Option<RunningBackend> {
        self.inner.lock().await.as_ref().map(|h| h.backend)
    }

    /// Stop any running server. Called before starting a new one and on app exit.
    pub async fn stop_all(&self) {
        let mut guard = self.inner.lock().await;
        if let Some(mut handle) = guard.take() {
            let backend = handle.backend;
            tracing::info!(%backend, "stopping backend server");
            // Ask the process to terminate gracefully first.
            let _ = handle.child.start_kill();
            // Give it a moment, then drop the handle (which also kills on drop
            // because we set kill_on_drop).
            let _ = tokio::time::timeout(Duration::from_secs(3), handle.child.wait()).await;
            tracing::info!(%backend, "backend server stopped");
        }
    }

    /// Start the Ollama server. If Ollama is already running, no-op; otherwise
    /// stop any other backend first. `binary` defaults to `ollama` on PATH.
    pub async fn start_ollama(&self, binary: Option<&str>) -> Result<RunningBackend> {
        if let Some(RunningBackend::Ollama) = self.running().await {
            return Ok(RunningBackend::Ollama);
        }
        self.stop_all().await;

        let program = binary.unwrap_or("ollama");
        let mut cmd = Self::base_command(program);
        cmd.arg("serve");
        let handle = self.spawn_tracked(cmd, RunningBackend::Ollama).await?;

        let url = "http://localhost:11434/api/tags";
        if let Err(e) = wait_for_url(url, Duration::from_secs(30)).await {
            tracing::warn!("Ollama health probe failed (continuing): {e}");
        }
        Ok(handle)
    }

    /// Build a detached command with piped stdout/stderr and null stdin.
    fn base_command(program: &str) -> Command {
        let mut cmd = if cfg!(target_os = "windows") {
            // On Windows, run detached via `cmd /C` so closing Phoenix's console
            // (if any) doesn't kill the child.
            let mut c = Command::new("cmd");
            c.arg("/C").arg(program);
            c
        } else {
            Command::new(program)
        };
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true);
        cmd
    }

    /// Spawn a command, spawn line-reader tasks for its stdout/stderr, store the
    /// handle, and return the backend kind.
    async fn spawn_tracked(
        &self,
        mut cmd: Command,
        backend: RunningBackend,
    ) -> Result<RunningBackend> {
        let mut child = cmd.spawn().map_err(|e| {
            PhoenixError::Other(format!("failed to spawn {backend:?}: {e}"))
        })?;

        // Pipe stdout/stderr to the log. Taking the pipes out of `child` lets the
        // process keep running while we read asynchronously; child.wait() still
        // works without them.
        if let Some(stdout) = child.stdout.take() {
            let backend_label = backend.as_str().to_string();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::info!(backend = %backend_label, "server: {line}");
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            let backend_label = backend.as_str().to_string();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::warn!(backend = %backend_label, "server: {line}");
                }
            });
        }

        tracing::info!(%backend, "backend server started");
        *self.inner.lock().await = Some(ServerHandle { child, backend });
        Ok(backend)
    }
}

/// Poll a URL until it responds 2xx or the timeout elapses.
async fn wait_for_url(url: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| PhoenixError::Other(e.to_string()))?;
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(PhoenixError::Other(format!(
                "{url} did not respond within {:?}",
                timeout
            )));
        }
        match client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => return Ok(()),
            _ => tokio::time::sleep(Duration::from_millis(400)).await,
        }
    }
}
