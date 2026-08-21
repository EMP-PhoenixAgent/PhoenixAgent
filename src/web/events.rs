//! Backend → frontend event forwarding.
//!
//! The agent runtime emits [`AgentEvent`]s and the health monitor emits
//! [`HealthState`] snapshots through tokio channels. These tasks drain those
//! channels and re-emit them as Tauri events so the webview can listen.

use tauri::{Emitter, Manager};
use tokio::sync::{mpsc, Mutex};

use super::state::WebState;
use crate::agent::AgentEvent;
use crate::health::HealthState;

/// Name of the Tauri event carrying agent events to the frontend.
pub const AGENT_EVENT: &str = "agent-event";
/// Name of the Tauri event carrying health updates to the frontend.
pub const HEALTH_EVENT: &str = "health-update";
/// Emitted whenever the active model/route changes (`set_model` + the `run_*`
/// commands) so the chat model selector and the Models panel stay in sync.
pub const MODEL_CHANGED_EVENT: &str = "model-changed";

/// Internal slots holding the receiver halves of the agent + health channels.
/// Populated by `unlock()` (commands.rs).
pub struct Forwarders {
    pub agent_rx: Mutex<Option<mpsc::Receiver<AgentEvent>>>,
    pub health_rx: Mutex<Option<mpsc::Receiver<HealthState>>>,
}

/// Run the forwarding tasks. Spawned once at app startup; waits until
/// `unlock()` has registered the [`Forwarders`], then drains channels forever.
pub async fn run_forwarders(app: tauri::AppHandle) {
    // Wait until the forwarders slot exists (set up by unlock()).
    loop {
        if app.try_state::<Forwarders>().is_some() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    // Spawn the agent event forwarder.
    {
        let app = app.clone();
        tokio::spawn(async move {
            loop {
                let event = {
                    let fwd = app.state::<Forwarders>();
                    let mut guard = fwd.agent_rx.lock().await;
                    match &mut *guard {
                        Some(rx) => rx.recv().await,
                        None => {
                            drop(guard);
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            continue;
                        }
                    }
                };
                if let Some(event) = event {
                    let _ = app.emit(AGENT_EVENT, event);
                }
            }
        });
    }

    // Spawn the health forwarder.
    {
        let app = app.clone();
        tokio::spawn(async move {
            loop {
                let snapshot = {
                    let fwd = app.state::<Forwarders>();
                    let mut guard = fwd.health_rx.lock().await;
                    match &mut *guard {
                        Some(rx) => rx.recv().await,
                        None => {
                            drop(guard);
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            continue;
                        }
                    }
                };
                if let Some(snapshot) = snapshot {
                    // Update the managed health state too.
                    if let Some(ws) = app.try_state::<WebState>() {
                        *ws.health.lock().await = snapshot.clone();
                    }
                    let _ = app.emit(HEALTH_EVENT, snapshot);
                }
            }
        });
    }
}
