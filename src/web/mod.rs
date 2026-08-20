//! Tauri integration layer — bridges the Rust backend to a web frontend.
//!
//! The web layer is a thin adapter over the existing [`crate::agent`] runtime
//! and [`crate::db`] store. It exposes Tauri commands the frontend calls via
//! `invoke()`, and forwards backend events/health to the webview via Tauri
//! `emit()`.

pub mod commands;
pub mod events;
pub mod model_urls;
pub mod state;

use std::sync::Arc;

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

    tauri::Builder::default()
        .manage(state)
        .setup(|app| {
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
            commands::setup,
            commands::unlock,
            commands::send_message,
            commands::context_resume,
            commands::learn,
            commands::set_mode,
            commands::get_mode,
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
            commands::list_models,
            commands::get_health,
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
            commands::run_ambercore,
            commands::connect_ambercore_remote,
            commands::use_local_ambercore,
            commands::get_ambercore_status,
            commands::list_ollama_models,
            commands::pull_ollama_model,
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
        ])
        .run(tauri::generate_context!())
        .map_err(|e| crate::error::PhoenixError::Other(format!("tauri: {e}")))?;

    Ok(())
}

/// Shared, thread-safe store handle used across the web layer.
pub(crate) type SharedStore = Arc<Mutex<crate::db::MemoryStore>>;
