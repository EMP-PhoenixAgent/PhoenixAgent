//! Tauri integration layer — bridges the Rust backend to a web frontend.
//!
//! The web layer is a thin adapter over the existing [`crate::agent`] runtime
//! and [`crate::db`] store. It exposes Tauri commands the frontend calls via
//! `invoke()`, and forwards backend events/health to the webview via Tauri
//! `emit()`.

pub mod commands;
mod logserver;
mod shard_pull;
pub mod events;
pub mod model_urls;
pub mod state;

use std::sync::Arc;

use tauri::Manager;

use tokio::sync::Mutex;

use crate::config::{Config, Paths};
use crate::error::Result;

/// Launch the Tauri GUI window.
///
/// Unlike the TUI path, this does NOT prompt for a passphrase here — the
/// passphrase is collected by the web unlock screen and passed back via the
/// `unlock` command.
pub fn run(config: Config, paths: Paths, workdir: std::path::PathBuf) -> Result<()> {
    let state = state::WebState::new(config, paths, workdir);

    // Session log: one JSON run file beside the exe, written live from launch
    // (pre-unlock lifecycle events included — Logs/ is outside the capsule).
    // Retention settings ride config.toml (sealed with the rest).
    {
        let hw = state.provider.embedded().state().hardware_status();
        let gpu = hw
            .gpu
            .as_ref()
            .map(|g| match g.vram_total_mb {
                Some(mb) => format!("{} · {} GB", g.name, mb / 1024),
                None => g.name.clone(),
            })
            .unwrap_or_else(|| "no GPU".into());
        let hardware = format!(
            "{gpu} · {} GB RAM · {}",
            hw.ram_total_mb.map(|mb| mb / 1024).unwrap_or(0),
            hw.backend,
        );
        let (auto_clean, keep_days, model) = {
            let c = state.config.blocking_lock();
            (c.logs_auto_clean, c.logs_keep_days, c.model.clone())
        };
        let header = crate::logsys::RunHeader {
            app: format!("Phoenix Agent {}", env!("CARGO_PKG_VERSION")),
            profile: "home".into(),
            workdir: state.workdir.blocking_lock().display().to_string(),
            hardware,
            models: vec![model],
            ..Default::default()
        };
        let report = crate::logsys::init(&state.paths.data_dir, header, auto_clean, keep_days);
        if report.first_creation {
            tracing::info!("logs: created {} beside the app", report.logs_dir);
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        // Encapsulated: a second copy of the exe would race the seal (and
        // the staging dir) — refuse to run concurrently.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.set_focus();
            }
        }))
        .manage(state)
        .setup(|app| {
            // Tap the AmberCore engine's lifecycle channel into the session
            // log — the SILENT recoveries (OOM CPU fallback, model loads/
            // unloads, VRAM verdicts) that never fail a request. The engine
            // exists from launch, so this drains from the very first event.
            {
                let mut rx = {
                    let ws = app.state::<state::WebState>();
                    ws.provider.embedded().state().subscribe()
                };
                tauri::async_runtime::spawn(async move {
                    loop {
                        match rx.recv().await {
                            Ok(ev) => crate::logsys::engine_event(&ev),
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                                tracing::warn!("logsys: engine event channel lagged ({n} missed)");
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        }
                    }
                });
            }
            // Spawn the background event/health forwarding tasks. They start
            // dormant (no agent runtime yet) and wake up after `unlock()`
            // stores the channels.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                events::run_forwarders(handle).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::is_initialized,
            commands::is_dev,
            commands::setup,
            commands::unlock,
            commands::send_message,
            commands::context_resume,
            commands::learn,
            commands::compact_context,
            commands::set_mode,
            commands::get_mode,
            // To-do list (shared plan panel)
            commands::get_todo,
            commands::set_todo,
            // Sub-Agents (Panel 6)
            commands::list_sub_agents,
            commands::create_sub_agent,
            commands::update_sub_agent,
            commands::delete_sub_agent,
            // Seven: alpha Chronos pop-up
            commands::should_show_alpha_popup,
            commands::dismiss_alpha_popup,
            commands::approve,
            commands::deny,
            commands::new_session,
            commands::list_sessions,
            commands::load_session_messages,
            commands::seal_capsule,
            commands::list_models,
            commands::get_health,
            commands::get_runtime_metrics,
            // GGUF inventory (Models panel)
            commands::scan_gguf_directory,
            commands::get_gguf_directory,
            commands::set_gguf_directory,
            // Models panel v0.5 — AmberCore / Ollama / Provider API
            commands::get_active_route,
            commands::list_ambercore_models,
            commands::set_ambercore_directory,
            commands::get_ambercore_directory,
            commands::pull_ambercore_model,
            commands::delete_ambercore_model,
            commands::run_ambercore,
            commands::connect_ambercore_remote,
            commands::use_local_ambercore,
            commands::get_ambercore_status,
            commands::get_hardware_status,
            commands::list_ollama_models,
            commands::pull_ollama_model,
            commands::delete_ollama_model,
            commands::install_ollama,
            commands::run_ollama,
            commands::list_providers,
            commands::create_provider,
            commands::update_provider,
            commands::delete_provider,
            commands::get_provider_key,
            commands::run_provider,
            commands::provider_usage_last_hour,
            // Science Workbench (Panel 1)
            commands::set_model,
            commands::get_backend,
            commands::set_backend,
            commands::list_profiles,
            commands::create_profile,
            commands::switch_profile,
            commands::get_workdir,
            commands::set_workdir,
            commands::console_run,
            commands::search_models,
            // Science Workbench (Panel 2: Skills)
            commands::list_skills,
            commands::list_skills_for_active_profile,
            commands::create_skill,
            commands::update_skill,
            commands::delete_skill,
            commands::set_skill_enabled,
            commands::search_github_skills,
            commands::install_github_skill,
            // Science Workbench (Panel 3: Tools)
            commands::list_tools,
            commands::list_tools_for_active_profile,
            commands::create_tool,
            commands::update_tool,
            commands::delete_tool,
            commands::set_tool_enabled,
            commands::prefetch_github_tool,
            commands::search_github_tools,
            // Science Workbench (Panel 4: Context)
            commands::list_context,
            commands::list_context_for_active_profile,
            commands::create_context,
            commands::update_context,
            commands::delete_context,
            commands::set_context_enabled,
            // Science Workbench (Panel 5: Memory / MCP)
            commands::list_memory,
            commands::list_memory_for_active_profile,
            commands::create_memory,
            commands::update_memory,
            commands::delete_memory,
            commands::set_memory_enabled,
            commands::test_memory_connection,
            // Security: two-password model (launch gate + DB password) + TOTP 2FA
            commands::has_launch_password,
            commands::set_launch_password,
            commands::recover_launch_via_totp,
            commands::has_totp,
            commands::setup_totp,
            commands::confirm_totp,
            commands::disable_totp,
            commands::change_passphrase,
            // Session logs (Logs/ beside the exe + LogsExplorer)
            commands::logs_settings_get,
            commands::logs_settings_set,
            commands::logs_stats,
            commands::logs_delete,
            commands::logs_open_explorer,
            commands::logs_reveal_folder,
        ])
        // Encapsulated exit path: on Exit, checkpoint the DB, stop the
        // backend processes, and seal the staging dir back INTO the exe
        // (rename-swap — safe from inside the dying process). This is the
        // moment the app folder becomes "nothing but the exe" again.
        .build(tauri::generate_context!())
        .map_err(|e| crate::error::PhoenixError::Other(format!("tauri: {e}")))?
        .run(|app, event| {
            if matches!(event, tauri::RunEvent::Exit) {
                // Finalize the session log BEFORE the seal: stamp `ended` and
                // rename `…` to the end hour. A crash never reaches here — the
                // open arrow left in the file name IS the crash marker.
                crate::logsys::finalize();
                let state = app.state::<crate::web::state::WebState>();
                match tauri::async_runtime::block_on(commands::seal_capsule_now(&state, true)) {
                    Ok(note) => tracing::info!("capsule: exit seal — {note}"),
                    Err(e) => tracing::error!("capsule: exit seal failed: {e} (staging dir keeps the state)"),
                }
            }
        });

    Ok(())
}

/// Shared, thread-safe store handle used across the web layer.
pub(crate) type SharedStore = Arc<Mutex<crate::db::MemoryStore>>;
