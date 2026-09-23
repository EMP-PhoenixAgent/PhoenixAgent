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

/// Mirror one agent event into the session log (metadata only — never text,
/// never tool outputs; see `crate::logsys`). This is the central tap: every
/// message, tool outcome, sub-agent and error the UI sees also lands in the
/// run file, with turn metrics pulled from the dispatch layer.
async fn log_agent_event(app: &tauri::AppHandle, event: &crate::agent::AgentEvent) {
    use crate::agent::AgentEvent;
    use crate::logsys::{self, By};
    match event {
        AgentEvent::AssistantMessage { tool_calls, reasoning, .. } => {
            if reasoning.is_some() {
                logsys::simple(By::Model, "Thinking");
            }
            let using = (!tool_calls.is_empty())
                .then(|| format!("{} tool call{}", tool_calls.len(), if tool_calls.len() == 1 { "" } else { "s" }));
            logsys::log(By::Model, "Message", using, None, true, None);
        }
        AgentEvent::ToolStarted { index, name, args } => {
            logsys::tool_started(*index, name, args);
        }
        AgentEvent::ToolNeedsApproval { index, name, args } => {
            // Write tools: the approval request precedes ToolStarted — capture
            // the signature now so a denial still logs what was asked.
            logsys::tool_started(*index, name, args);
            logsys::simple(By::Agent, "Validation");
        }
        AgentEvent::ToolFinished { index, name, success, result, duration_ms } => {
            logsys::tool_finished(*index, name, *success, result, *duration_ms);
        }
        AgentEvent::ToolDenied { index, name } => {
            logsys::tool_denied(*index, name);
        }
        AgentEvent::SubAgentStarted { name, model, .. } | AgentEvent::SubAgentFinished { name, model, .. } => {
            logsys::log(By::Agent, "SubAgent", Some(format!("{name} · {model}")), None, true, None);
        }
        AgentEvent::TurnDone { .. } => {
            if let Some(ws) = app.try_state::<WebState>() {
                let st = ws.provider.merged_stats().await;
                logsys::log(
                    By::Model,
                    "Turn",
                    None,
                    Some(logsys::Perfs {
                        tps: st.tokens_per_sec,
                        ttft_ms: st.ttft_ms,
                        tbt_ms: st.tbt_avg_ms,
                        dur_ms: None,
                    }),
                    true,
                    None,
                );
            }
        }
        AgentEvent::Error { message } => {
            logsys::error_now("Error", None, message);
        }
        _ => {}
    }
}

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
                    log_agent_event(&app, &event).await;
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
