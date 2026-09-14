//! Tauri command handlers — the API surface the frontend calls via `invoke()`.

use std::sync::Arc;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::sync::{watch, Mutex};

use super::events::{Forwarders, MODEL_CHANGED_EVENT};
use super::state::WebState;
use crate::agent::runtime::{McpConnectionSpec, ToolSpec};
use crate::agent::skills as skill_helpers;
use crate::agent::mcp as mcp_helpers;
use crate::agent::{AgentRuntime, Command, GithubSkillHit};
use crate::config::{save_config, ApprovalPolicy};
use crate::crypto::totp as totp_lib;
use crate::crypto::{
    derive_key, load_or_create_salt, random_key, rotate_salt, DerivedKey, KeyBundle, KEY_LEN,
};
use crate::db::{
    open_encrypted, ContextFile, MemoryStore, Profile, Provider, ProfileContext, ProfileSkill,
    ProfileTool, SessionSummary, Skill, ToolRow,
};
use crate::health;
use crate::model::dispatch::{ActiveRoute, CloudRoute, LocalBackend};
use crate::model::ModelProvider;
use crate::model::{ChatEvent, ChatMessage, ChatRequest, ChatRole};

/// Result of a successful unlock/setup — tells the frontend what to show.
#[derive(Debug, Serialize)]
pub struct UnlockResult {
    pub model: String,
    pub project_path: String,
    pub active_profile: Option<Profile>,
}

// ===== Encapsulated: capsule key plumbing ======================================
//
// v2 capsules are encrypted under a random capsule key that lives ONLY
// inside the key bundle's wraps. The bundle is read from staging (`keys.phx`)
// when it exists — and, because a v2 exit wipes staging (the exe carries
// everything), from the exe's own tail header when it doesn't. Both copies
// are identical: every bundle save updates staging, and every seal re-embeds
// the current staging bundle as the new tail header.

use zeroize::Zeroizing;

/// The current exe path, or a descriptive error.
fn current_exe() -> Result<std::path::PathBuf, String> {
    std::env::current_exe().map_err(|e| format!("current exe: {e}"))
}

/// Load the key bundle from staging, falling back to the v2 capsule tail
/// header (pre-hydration, when staging is empty). `Ok(None)` = no bundle
/// anywhere (a fresh install).
fn load_bundle_anywhere(state: &WebState) -> Result<Option<KeyBundle>, String> {
    if state.paths.key_bundle_path.exists() {
        return KeyBundle::load(&state.paths.key_bundle_path)
            .map(Some)
            .map_err(|e| format!("Read key bundle: {e}"));
    }
    if let Ok(Some(header)) = crate::capsule::read_header(&current_exe()?) {
        let text = String::from_utf8(header).map_err(|e| format!("capsule header: {e}"))?;
        return KeyBundle::parse(&text)
            .map(Some)
            .map_err(|e| format!("capsule header bundle: {e}"));
    }
    Ok(None)
}

/// Hydrate staging from the exe's capsule (missing files only). For a v2
/// capsule this is the ONLY hydration point — the tail is ciphertext until
/// the key arrives (post-gate). Errors propagate: a partial hydration must
/// not silently fall through to "fresh install" (which would re-create an
/// empty DB over the user's sealed state).
fn hydrate_capsule(state: &WebState, capsule_key: &[u8; KEY_LEN]) -> Result<usize, String> {
    let exe = current_exe()?;
    match crate::capsule::Capsule::open(&exe, Some(capsule_key)) {
        Ok(Some(cap)) => cap
            .extract_missing(&exe, &state.paths.data_dir)
            .map_err(|e| format!("capsule hydration: {e}")),
        Ok(None) => Ok(0),
        Err(e) => Err(format!("capsule open: {e}")),
    }
}

/// Whether the encrypted DB exists yet (controls setup vs. unlock screen).
/// A sealed (v2) capsule carries the DB even when staging was wiped at the
/// last exit — the unlock gate must still show.
#[tauri::command]
pub async fn is_initialized(state: State<'_, WebState>) -> Result<bool, String> {
    if state.paths.db_path.exists() {
        return Ok(true);
    }
    if let Ok(exe) = current_exe() {
        if matches!(
            crate::capsule::probe(&exe),
            Ok(Some(crate::capsule::CapsuleVer::V2))
        ) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether the app was built in debug/dev mode. The frontend uses this to
/// auto-unlock during `cargo tauri dev` so iteration doesn't require retyping
/// the launch password. Stripped from release builds (`cfg!(debug_assertions)`
/// is false in release).
#[tauri::command]
pub async fn is_dev() -> bool {
    cfg!(debug_assertions)
}

/// The default "database access password" — the secret that derives the
/// SQLCipher key. Set once at first run and then used autonomously by the app;
/// the user never has to type it again (it lives wrapped under their launch
/// password). Used as a last-resort recovery code if the launch password is
/// lost and 2FA is not enabled.
pub const DEFAULT_DB_PASSWORD: &str = "PhoenixAgent";

/// First-run setup: create the encrypted DB and wrap its key under a launch
/// password.
///
/// The user chooses a **launch password** (typed at every startup thereafter).
/// The SQLCipher DB key is derived from the fixed [`DEFAULT_DB_PASSWORD`]
/// ("PhoenixAgent"); that key is then **wrapped** under the launch password and
/// stored in `keys.phx`, so the user never has to remember the DB password —
/// the app unwraps it autonomously on each launch.
#[tauri::command]
pub async fn setup(
    app: AppHandle,
    state: State<'_, WebState>,
    launch_password: String,
    confirm_launch_password: String,
    model: Option<String>,
) -> Result<UnlockResult, String> {
    // 1. Validate the launch password.
    if launch_password.len() < 8 {
        return Err("Launch password must be at least 8 characters.".into());
    }
    if launch_password != confirm_launch_password {
        return Err("Launch passwords do not match.".into());
    }

    // 2. Update config model if provided, then persist.
    if let Some(model) = &model {
        let mut cfg = state.config.lock().await;
        cfg.model = model.clone();
        let cfg_clone = cfg.clone();
        drop(cfg);
        save_config(&state.paths, &cfg_clone).map_err(|e| format!("Save config: {e}"))?;
    }

    // 3. Derive the DB key from the fixed default DB password + salt, and create
    //    the encrypted DB. The DB password is an internal anchor the user never
    //    types again.
    let salt = load_or_create_salt(&state.paths.salt_path)
        .map_err(|e| format!("Salt error: {e}"))?;
    let db_key = derive_key(DEFAULT_DB_PASSWORD, &salt, None)
        .map_err(|e| format!("Key derivation error: {e}"))?;
    let conn = open_encrypted(&state.paths.db_path, &db_key)
        .map_err(|e| format!("Failed to create database: {e}"))?;

    // 4. Wrap the DB key AND a fresh capsule key (encrypts the exe's sealed
    //    state tail — see capsule.rs) under the launch password and persist
    //    the bundle. From this install's first seal on, everything the exe
    //    carries is ciphertext.
    let capsule_key = random_key();
    let bundle = KeyBundle::create(&db_key, &launch_password, Some(&capsule_key))
        .map_err(|e| format!("Wrap DB key: {e}"))?;
    bundle
        .save(&state.paths.key_bundle_path)
        .map_err(|e| format!("Save key bundle: {e}"))?;
    *state.capsule_key.lock().await = Some(Zeroizing::new(capsule_key));

    // 5. Boot the runtime (shared with unlock).
    boot_runtime(app, state, conn, db_key).await
}

/// Unlock by typing the **launch password**, which unwraps the DB key from the
/// on-disk key bundle. The DB password is never asked for on a normal launch —
/// the app uses it autonomously once unwrapped.
/// Encapsulated: the same password also unwraps the **capsule key** (the
/// bundle may come from the exe's tail header — staging can be empty after a
/// wiped exit). A v2 capsule hydrates ONLY here, behind the gate. A v1 exe
/// (pre-Phase-C plaintext capsule, or no capsule wrap yet) MINTS a capsule
/// key wrapped under this password — the migration; its first seal after
/// this turn re-encrypts the whole tail.
#[tauri::command]
pub async fn unlock(
    app: AppHandle,
    state: State<'_, WebState>,
    launch_password: String,
) -> Result<UnlockResult, String> {
    let mut bundle = load_bundle_anywhere(&state)?
        .ok_or("No key bundle found — the app is not initialized.")?;

    // Capsule key: unwrap (v2) or mint+wrap (v1 migration / fresh install).
    let capsule_key = if bundle.has_capsule_key() {
        bundle
            .unwrap_capsule_key(&launch_password)
            .map_err(|_| "Wrong launch password.".to_string())?
    } else {
        let k = random_key();
        bundle
            .set_capsule_key(&k, &launch_password)
            .map_err(|e| format!("Capsule wrap: {e}"))?;
        bundle
            .save(&state.paths.key_bundle_path)
            .map_err(|e| format!("Save key bundle: {e}"))?;
        k
    };
    // Hydrate staging from the exe (v2: decrypts only now; v1: already
    // hydrated at boot, so this no-ops). Fail hard — see hydrate_capsule.
    let hydrated = hydrate_capsule(&state, &capsule_key)?;
    if hydrated > 0 {
        tracing::info!("capsule: hydrated {hydrated} file(s) from the exe");
    }
    *state.capsule_key.lock().await = Some(Zeroizing::new(capsule_key));

    let db_key = bundle
        .unwrap_primary(&launch_password)
        .map_err(|_| "Wrong launch password.".to_string())?;
    let conn = open_encrypted(&state.paths.db_path, &db_key)
        .map_err(|e| format!("Database error after unwrap: {e}"))?;

    boot_runtime(app, state, conn, db_key).await
}

/// Whether a launch password is set (i.e. a key bundle exists with a primary
/// wrap). Callable before unlock so the startup screen knows to show the
/// launch gate. Reads the exe tail header when staging has no bundle (a
/// wiped v2 exit).
#[tauri::command]
pub async fn has_launch_password(state: State<'_, WebState>) -> Result<bool, String> {
    Ok(load_bundle_anywhere(&state)?.is_some())
}

/// Change the launch password. Requires the current launch password to prove
/// ownership; re-wraps the DB key under the new password. Does NOT rekey the
/// DB (the SQLCipher key is untouched), so it's instant and risk-free.
#[tauri::command]
pub async fn set_launch_password(
    state: State<'_, WebState>,
    current_password: String,
    new_password: String,
    confirm: String,
) -> Result<(), String> {
    if new_password.len() < 8 {
        return Err("New launch password must be at least 8 characters.".into());
    }
    if new_password != confirm {
        return Err("New launch passwords do not match.".into());
    }
    let mut bundle = load_bundle_anywhere(&state)?
        .ok_or("No key bundle found — the app is not initialized.")?;
    // Prove ownership by unwrapping BOTH wraps with the current password.
    let db_key = bundle
        .unwrap_primary(&current_password)
        .map_err(|_| "Current launch password is incorrect.".to_string())?;
    let capsule_key = if bundle.has_capsule_key() {
        Some(
            bundle
                .unwrap_capsule_key(&current_password)
                .map_err(|_| "Current launch password is incorrect.".to_string())?,
        )
    } else {
        None
    };
    bundle
        .change_primary(&db_key, capsule_key.as_ref(), &new_password)
        .map_err(|e| format!("Re-wrap: {e}"))?;
    bundle
        .save(&state.paths.key_bundle_path)
        .map_err(|e| format!("Save key bundle: {e}"))?;
    Ok(())
}

/// Recover access when the launch password is forgotten, using a current TOTP
/// code (only works if 2FA is enabled, which created the recovery wrap).
///
/// On success this verifies the code against the stored recovery seed, unwraps
/// the DB key via the recovery wrap, **sets a new launch password** (mandatory
/// — recovering resets the gate), re-wraps the key under it, and boots the
/// runtime, returning the same `UnlockResult` as `unlock` so the UI transitions
/// to the chat screen. On a wrong code it returns `Err` with a generic message.
#[tauri::command]
pub async fn recover_launch_via_totp(
    app: AppHandle,
    state: State<'_, WebState>,
    totp_code: String,
    new_launch_password: String,
) -> Result<UnlockResult, String> {
    if new_launch_password.len() < 8 {
        return Err("New launch password must be at least 8 characters.".into());
    }
    let mut bundle = load_bundle_anywhere(&state)?
        .ok_or("No key bundle found — the app is not initialized.")?;
    if !bundle.has_recovery() {
        return Err("Recovery is not available: 2FA is not enabled.".into());
    }
    // Verify the typed code against the stored recovery seed.
    if bundle.verify_recovery_code(&totp_code).is_err() {
        return Err("Recovery code is incorrect.".into());
    }
    // Unwrap the DB key via the recovery wrap.
    let db_key = bundle
        .unwrap_recovery()
        .map_err(|_| "Recovery unwrap failed.".to_string())?;
    // The capsule key recovers the same way (a v1 exe without a capsule wrap
    // mints one here — the recovery-path migration).
    let capsule_key = if bundle.has_capsule_key() {
        bundle
            .unwrap_capsule_recovery()
            .map_err(|_| "Recovery unwrap failed.".to_string())?
    } else {
        random_key()
    };
    // Hydrate BEFORE opening the DB — a wiped v2 exit leaves staging empty.
    let hydrated = hydrate_capsule(&state, &capsule_key)?;
    if hydrated > 0 {
        tracing::info!("capsule: hydrated {hydrated} file(s) from the exe");
    }
    *state.capsule_key.lock().await = Some(Zeroizing::new(capsule_key));
    // Set a new launch password (re-wrap both keys under it).
    bundle
        .change_primary(&db_key, Some(&capsule_key), &new_launch_password)
        .map_err(|e| format!("Re-wrap after recovery: {e}"))?;
    bundle
        .save(&state.paths.key_bundle_path)
        .map_err(|e| format!("Save key bundle: {e}"))?;
    // Open the DB and boot the runtime with the recovered key.
    let conn = open_encrypted(&state.paths.db_path, &db_key)
        .map_err(|e| format!("Database error after recovery: {e}"))?;
    boot_runtime(app, state, conn, db_key).await
}

/// Shared runtime boot: wraps the store, ensures a default profile, loads the
/// active profile + persisted workdir, spawns health monitor + agent runtime,
/// stores channels, and registers event forwarders. Used by both `setup` and
/// `unlock` so the post-DB-open logic isn't duplicated.
async fn boot_runtime(
    app: AppHandle,
    state: State<'_, WebState>,
    conn: rusqlite::Connection,
    db_key: DerivedKey,
) -> Result<UnlockResult, String> {
    // Retain the DB key in memory while unlocked (needed to maintain the 2FA
    // recovery wrap and DB-rekey re-wrap). Zeroized on drop.
    *state.db_key.lock().await = Some(db_key);
    // Wrap the store and ensure a default profile exists.
    let store: super::SharedStore = Arc::new(Mutex::new(MemoryStore::new(conn)));
    {
        let s = store.lock().await;
        s.ensure_default_profile().map_err(|e| e.to_string())?;
        // Seed starter skills if the table is empty (first run / new install).
        s.ensure_seed_skills().map_err(|e| e.to_string())?;
        // Seed starter tools (scientific/teaching utilities) if the tools table
        // is empty.
        s.ensure_seed_tools().map_err(|e| e.to_string())?;
    }

    // Resolve the active profile (fall back to the default inside the store).
    let active_profile = {
        let s = store.lock().await;
        let id = s.get_active_profile_id().map_err(|e| e.to_string())?;
        match id {
            Some(id) => s.get_profile(id).map_err(|e| e.to_string())?,
            None => None,
        }
    };

    // Build the effective config: base config.toml, with the active profile's
    // behavior settings overriding model-independent fields.
    let mut cfg = state.config.lock().await.clone();
    if let Some(p) = &active_profile {
        cfg.approval_policy = parse_approval_policy(&p.approval_policy);
        cfg.max_iterations = p.max_iterations as u32;
        cfg.context_window = p.context_window as u32;
    }

    // Resolve the working directory: the ACTIVE PROFILE's own directory first
    // (workdir is profile-scoped), then the legacy global setting, else the
    // launch workdir. Update WebState so commands see the resolved value.
    let workdir = {
        let s = store.lock().await;
        let profile_wd = s
            .get_active_profile_id()
            .ok()
            .flatten()
            .and_then(|id| s.get_profile_workdir(id).ok().flatten())
            .filter(|p| !p.is_empty());
        match profile_wd.or_else(|| s.get_workdir().ok().flatten().filter(|p| !p.is_empty())) {
            Some(p) => std::path::PathBuf::from(p),
            None => state.workdir.lock().await.clone(),
        }
    };
    *state.workdir.lock().await = workdir.clone();
    let project_path = workdir.display().to_string();

    // Load the skills enabled for the active profile (for prompt seeding).
    let enabled_skills: Vec<(String, String, String)> = {
        let s = store.lock().await;
        match &active_profile {
            Some(p) => s
                .list_enabled_skills_for_profile(p.id)
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(|sk| (sk.name, sk.description, sk.body))
                .collect(),
            None => Vec::new(),
        }
    };

    // Load the user tools enabled for the active profile (registry seeding).
    let enabled_user_tools: Vec<ToolSpec> = {
        let s = store.lock().await;
        match &active_profile {
            Some(p) => ToolSpec::from_rows(
                &s.list_enabled_tools_for_profile(p.id)
                    .map_err(|e| e.to_string())?,
            ),
            None => Vec::new(),
        }
    };

    // Load the context files enabled for the active profile (prompt seeding).
    let enabled_context: Vec<(String, String, String)> = {
        let s = store.lock().await;
        match &active_profile {
            Some(p) => s
                .list_enabled_context_for_profile(p.id)
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(|c| (c.name, c.description, c.body))
                .collect(),
            None => Vec::new(),
        }
    };

    // Load the MCP connections enabled for the active profile (tool seeding).
    let enabled_mcp: Vec<McpConnectionSpec> = {
        let s = store.lock().await;
        match &active_profile {
            Some(p) => s
                .list_enabled_memory_for_profile(p.id)
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(|m| McpConnectionSpec {
                    id: m.id,
                    name: m.name,
                    transport: m.transport,
                    command: m.command,
                    args_json: m.args_json,
                })
                .collect(),
            None => Vec::new(),
        }
    };

    // Load the persisted to-do list (the shared plan) so the system prompt and
    // the UI panel start from where the last session left off.
    let todo: String = {
        let s = store.lock().await;
        s.get_setting("todo_markdown")
            .map_err(|e| e.to_string())?
            .unwrap_or_default()
    };

    let model_name = cfg.model.clone();

    // If a cloud provider was active at last shutdown, re-apply its endpoint now
    // that the DB is open (we can finally read the API key). This keeps a cloud
    // selection persistent across restarts. If the provider no longer exists,
    // clear the stale selection.
    if let Some(pid) = cfg.active_provider_id {
        let endpoint = store.lock().await.get_provider_endpoint(pid);
        match endpoint {
            Ok(Some((base_url, key))) => {
                state
                    .provider
                    .set_cloud(CloudRoute {
                        provider_id: pid,
                        base_url,
                        api_key: key,
                    })
                    .await;
                tracing::info!(provider_id = pid, "re-applied active cloud provider");
            }
            _ => {
                tracing::warn!(provider_id = pid, "active provider no longer exists; reverting to local");
                let mut c = state.config.lock().await;
                c.active_provider_id = None;
                if let Err(e) = save_config(&state.paths, &c) {
                    tracing::error!("clearing stale active_provider_id failed: {e}");
                }
                cfg.active_provider_id = None;
            }
        }
    }

    // ---- One-time backend resolution (boot) ----
    // Local AmberCore now means the **embedded in-process engine** — always
    // available (models load lazily), so it needs no HTTP probe: activate it
    // directly and warm the last-used model. Remote AmberCore and Ollama keep
    // the reachability probe below. Skipped when a cloud provider is active.
    let mut warm_embedded_model = false;
    if cfg.active_provider_id.is_none() {
        if cfg.active_backend == "ambercore" && !cfg.ambercore_remote {
            state.provider.set_local_embedded().await;
            warm_embedded_model = true;
            tracing::info!("backend: embedded AmberCore (in-process) active at launch");
        } else {
        let ollama_url = cfg.ollama_url.clone();
        let ambercore_url = cfg.ambercore_url.clone();
        let configured = cfg.active_backend.clone();
        let probe = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(1500))
            .build();
        if let Ok(client) = probe {
            let (ollama_up, amber_up) = tokio::join!(
                endpoint_reachable(&client, &ollama_url),
                endpoint_reachable(&client, &ambercore_url),
            );
            // Prefer the configured backend when it's up; else take whichever is.
            let detected: Option<&str> = match (configured.as_str(), ollama_up, amber_up) {
                ("ambercore", _, true) => Some("ambercore"),
                ("ollama", true, _) => Some("ollama"),
                (_, true, false) => Some("ollama"),
                (_, false, true) => Some("ambercore"),
                _ => None,
            };
            match detected {
                Some(b) if b != configured.as_str() => {
                    let new_url = if b == "ambercore" { ambercore_url.clone() } else { ollama_url.clone() };
                    let lb = LocalBackend::parse(b).unwrap_or(LocalBackend::Ollama);
                    state.provider.set_local(lb, new_url).await;
                    state.config.lock().await.active_backend = b.to_string();
                    tracing::info!(
                        "backend auto-detect: switched to {b} (ollama_up={ollama_up}, ambercore_up={amber_up})"
                    );
                }
                Some(b) => {
                    tracing::info!(
                        "backend auto-detect: kept {b} (ollama_up={ollama_up}, ambercore_up={amber_up})"
                    );
                }
                None => {
                    tracing::info!(
                        "backend auto-detect: none reachable, keeping {configured} (ollama_up={ollama_up}, amber_up={amber_up})"
                    );
                }
            }
        }
        } // else: remote AmberCore / Ollama reachability probe
    }

    // Watch channel for the live model name (shared by runtime + health).
    let (model_tx, model_rx) = watch::channel(model_name.clone());
    *state.model_tx.lock().await = Some(model_tx);

    // Spawn health monitor (subscribed to live model changes).
    let health_rx = health::spawn_monitor(
        state.provider_dyn(),
        model_rx,
        store.clone(),
        5,
    );

    // Spawn the agent runtime (behavior seeded from the active profile; skills,
    // user tools, and context files seeded from the profile's enabled sets).
    let handle = AgentRuntime::spawn(
        state.provider_dyn(),
        model_name.clone(),
        cfg,
        workdir,
        store.clone(),
        enabled_skills,
        enabled_user_tools,
        enabled_context,
        enabled_mcp,
        todo,
    );

    // Store the channels for commands to use.
    *state.cmd_tx.lock().await = Some(handle.cmd_tx);
    *state.store.lock().await = Some(store);

    // Register the forwarders so events/health flow to the webview.
    app.manage(Forwarders {
        agent_rx: Mutex::new(Some(handle.event_rx)),
        health_rx: Mutex::new(Some(health_rx)),
    });

    // Launch-time warm-up: preload the last-used model into the embedded
    // AmberCore engine's pool (background task — loads the GGUF so the first
    // message doesn't pay the cold-start). This is what makes "launching the
    // agent also launches AmberCore with the latest used model" true.
    if warm_embedded_model {
        let provider = state.provider.clone();
        let model = model_name.clone();
        let app = app.clone();
        tokio::spawn(async move {
            provider.warm_ambercore_model(&model).await;
            emit_vram_status(&app, &provider);
        });
    }

    Ok(UnlockResult { model: model_name, project_path, active_profile })
}

/// Parse a stored profile policy string back into the enum. Falls back to the
/// default (`WritesOnly`) for anything unrecognized.
fn parse_approval_policy(s: &str) -> ApprovalPolicy {
    match s {
        "all" => ApprovalPolicy::All,
        "never" => ApprovalPolicy::Never,
        _ => ApprovalPolicy::WritesOnly,
    }
}

/// Send a user message to the agent.
#[tauri::command]
pub async fn send_message(state: State<'_, WebState>, text: String) -> Result<(), String> {
    let tx = state.cmd_tx.lock().await;
    match tx.as_ref() {
        Some(sender) => sender
            .send(Command::UserInput { text })
            .await
            .map_err(|e| format!("Agent channel closed: {e}")),
        None => Err("Not unlocked yet".into()),
    }
}

/// `/context-resume` — gather the active profile + its enabled context files and
/// ask the agent (as a normal user turn) to produce a full project résumé. The
/// résumé streams into chat like any assistant reply. Sent as a `UserInput` with
/// a rich prompt so the model has the context bodies in-context even if the
/// system prompt's context section was trimmed.
#[tauri::command]
pub async fn context_resume(state: State<'_, WebState>) -> Result<(), String> {
    let pid = active_profile_id(&state).await?;
    let (profile_name, context_files): (String, Vec<ProfileContext>) = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        let name = s
            .list_profiles()
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|p| p.id == pid)
            .map(|p| p.name)
            .unwrap_or_else(|| "(unnamed)".to_string());
        let ctxs = s.list_context_for_profile(pid).map_err(|e| e.to_string())?;
        (name, ctxs)
    };
    let workdir = state.workdir.lock().await.display().to_string();

    let enabled: Vec<&ProfileContext> = context_files.iter().filter(|c| c.enabled).collect();
    let mut prompt = format!(
        "Produce a comprehensive résumé of the current project for the user.\n\n\
         Active profile: {profile_name}\nWorking directory: {workdir}\n"
    );
    if enabled.is_empty() {
        prompt.push_str(
            "\nNo context files are enabled for this profile — résumé what you can \
             from the conversation so far and the working directory.",
        );
    } else {
        prompt.push_str("\nEnabled project context files:");
        for c in &enabled {
            // Cap each body so the prompt can't grow unbounded.
            let body: String = c.context.body.chars().take(4000).collect();
            prompt.push_str(&format!(
                "\n\n### {}\n_{}_\n{}",
                c.context.name, c.context.description, body
            ));
        }
        prompt.push_str(
            "\n\nSummarize the project: its purpose, structure, conventions, key \
             decisions, and current state. Be thorough but readable.",
        );
    }

    let tx = state.cmd_tx.lock().await;
    match tx.as_ref() {
        Some(sender) => sender
            .send(Command::UserInput { text: prompt })
            .await
            .map_err(|e| format!("Agent channel closed: {e}")),
        None => Err("Not unlocked yet".into()),
    }
}

/// `/learn` — update the agent's PROJECT MEMORY: one living context file,
/// auto-maintained, holding what the agent has to do (open tasks,
/// requirements, standing constraints, user preferences) and what it has
/// done (completed work, decisions, outcomes). The previous memory note is
/// merged with the current conversation into an updated note written back
/// to the SAME file — the memory stays one compact note instead of piling
/// up per-run copies. Distinct from `/cc`: /cc snapshots the conversation
/// into a new timestamped compact and rotates the session to free the
/// context window; /learn is memory management and keeps the session
/// running. Uses Phoenix's native context-file memory (no external MCP).
#[tauri::command]
pub async fn learn(state: State<'_, WebState>) -> Result<String, String> {
    // Fixed name so the memory file is found and updated in place every run.
    const MEMORY_NAME: &str = "Project Memory";
    const MEMORY_DESC: &str = "Auto-maintained by /learn — the agent's project memory: \
                               what has to be done and what has been done.";

    // 1. Most-recent session = the conversation to learn from. Lock config +
    //    store separately (never nested) to avoid lock-order deadlocks.
    let model = state.config.lock().await.model.clone();
    let session_id = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.list_sessions()
            .map_err(|e| e.to_string())?
            .into_iter()
            .next()
            .ok_or("No conversation yet — nothing to learn.")?
            .id
    };

    // 2. Build the transcript (user/assistant turns). Full session: anything
    //    older is already carried inside the previous memory note, which is
    //    merged in below.
    let messages = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.load_messages(session_id, u32::MAX)
            .map_err(|e| e.to_string())?
    };
    let mut transcript = String::new();
    for m in &messages {
        match m.role {
            ChatRole::User => transcript.push_str(&format!("User: {}\n", m.content)),
            ChatRole::Assistant => transcript.push_str(&format!("Assistant: {}\n", m.content)),
            _ => {}
        }
    }
    if transcript.trim().is_empty() {
        return Err("No conversation to learn from yet.".into());
    }

    // 3. The existing memory note (fixed-name file), if this isn't the first
    //    /learn run.
    let pid = active_profile_id(&state).await?;
    let previous = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.list_context_for_profile(pid)
            .map_err(|e| e.to_string())?
            .into_iter()
            .find(|pc| pc.context.name == MEMORY_NAME)
            .map(|pc| pc.context)
    };
    let old_body = previous
        .as_ref()
        .map(|c| c.body.trim().to_string())
        .unwrap_or_default();

    // 4. One-shot merge (no tools, no chat echo): previous memory + new
    //    conversation -> updated memory.
    let request = ChatRequest {
        messages: vec![
            ChatMessage::system(
                "You maintain an AI agent's project memory — one dense note the agent \
                 re-reads every session. Merge the previous memory with the new conversation \
                 into the UPDATED memory. It must answer two questions: (1) WHAT HAS TO BE \
                 DONE — open tasks, bugs, requirements, standing constraints and user \
                 preferences; (2) WHAT HAS BEEN DONE — completed work, decisions with their \
                 rationale, and the current state of each effort. Carry forward anything \
                 still relevant, drop what is obsolete or superseded, add what is new. Be \
                 specific — names, paths, numbers. Output ONLY the updated memory in \
                 markdown, no commentary.",
            ),
            ChatMessage::user(format!(
                "# Previous project memory\n{}\n\n# New conversation transcript\n{transcript}\n\n# Task\nWrite the updated project memory now.",
                if old_body.is_empty() { "(none yet)" } else { &old_body },
            )),
        ],
        tools: Vec::new(),
        temperature: 0.2,
    };
    let mut rx = state
        .provider
        .chat(&model, request)
        .await
        .map_err(|e| e.to_string())?;
    let mut note = String::new();
    while let Some(ev) = rx.recv().await {
        if let ChatEvent::Delta(d) = ev {
            note.push_str(&d);
        }
    }
    if note.trim().is_empty() {
        return Err("The model returned an empty memory note — nothing to save.".into());
    }

    // 5. Write back to the SAME file (created on first run) and keep it
    //    enabled for the active profile.
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        let id = match &previous {
            Some(ctx) => {
                s.update_context(ctx.id, MEMORY_NAME, MEMORY_DESC, &note)
                    .map_err(|e| e.to_string())?;
                ctx.id
            }
            None => s
                .create_context(MEMORY_NAME, MEMORY_DESC, &note)
                .map_err(|e| e.to_string())?,
        };
        s.set_context_enabled_for_profile(pid, id, true)
            .map_err(|e| e.to_string())?;
        // Stamp the session for the history-rail colors (learned → orange/green).
        s.mark_session_learned(session_id).map_err(|e| e.to_string())?;
    }
    reload_context(&state).await?;

    Ok(if previous.is_some() {
        "Project memory updated — what has to be done and what has been done, re-injected each session."
    } else {
        "Project memory created and enabled — what has to be done and what has been done, re-injected each session."
    }
    .to_string())
}

/// `/cc` — Context Compacting: snapshot the WHOLE current conversation (via
/// the active model) into a new timestamped context file capturing what has
/// been done, what is going to happen next and the topics under discussion,
/// then rotate to a fresh session so the long history is replaced by the
/// compact (freeing the context window). The old session stays in the DB —
/// nothing is deleted, just no longer loaded. The compacting counterpart to
/// `/learn`, which maintains the agent's living project memory instead and
/// keeps the session running.
#[tauri::command]
pub async fn compact_context(state: State<'_, WebState>) -> Result<String, String> {
    let model = state.config.lock().await.model.clone();
    let session_id = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.list_sessions()
            .map_err(|e| e.to_string())?
            .into_iter()
            .next()
            .ok_or("No conversation yet — nothing to compact.")?
            .id
    };

    // Full history (u32::MAX = no practical cap), not /learn's last-50.
    let messages = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.load_messages(session_id, u32::MAX)
            .map_err(|e| e.to_string())?
    };
    let mut transcript = String::new();
    for m in &messages {
        match m.role {
            ChatRole::User => transcript.push_str(&format!("User: {}\n", m.content)),
            ChatRole::Assistant => transcript.push_str(&format!("Assistant: {}\n", m.content)),
            _ => {}
        }
    }
    if transcript.trim().is_empty() {
        return Err("No conversation to compact yet.".into());
    }

    // One-shot compaction (no tools, no chat echo) — tuned to carry work
    // state, not just facts: the compact REPLACES the history it summarizes.
    let request = ChatRequest {
        messages: vec![
            ChatMessage::system(
                "You are a context compactor. The conversation below is being archived and \
                 replaced by your summary — the agent continues from your note alone, in a \
                 fresh session. Capture, in dense markdown: (1) WHAT HAS BEEN DONE in this \
                 conversation — work completed, decisions with their rationale, errors \
                 encountered and their fixes, and every file path, command, model or setting \
                 used or changed; (2) WHAT IS GOING TO HAPPEN NEXT — planned work, pending \
                 items and the concrete next steps; (3) THE TOPICS under discussion between \
                 the user and the agent — the subject threads and directions the user is \
                 following — so the next session picks up exactly where this one left off. \
                 Be specific — names, paths, numbers — not generic. Output ONLY the note.",
            ),
            ChatMessage::user(format!(
                "# Conversation transcript\n{transcript}\n\n# Task\nWrite the compact note now."
            )),
        ],
        tools: Vec::new(),
        temperature: 0.2,
    };
    let mut rx = state
        .provider
        .chat(&model, request)
        .await
        .map_err(|e| e.to_string())?;
    let mut note = String::new();
    while let Some(ev) = rx.recv().await {
        if let ChatEvent::Delta(d) = ev {
            note.push_str(&d);
        }
    }
    if note.trim().is_empty() {
        return Err("The model returned an empty compact — nothing to save.".into());
    }

    // New timestamped context file (kept each run, per user preference),
    // enabled for the active profile and re-injected into the system prompt.
    let when = chrono::Local::now().format("%Y-%m-%d %H:%M").to_string();
    let name = format!("Context Compact — {when}");
    let pid = active_profile_id(&state).await?;
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        let id = s
            .create_context(
                &name,
                "Auto-saved by /cc — compacted conversation replacing the prior session's history.",
                &note,
            )
            .map_err(|e| e.to_string())?;
        s.set_context_enabled_for_profile(pid, id, true)
            .map_err(|e| e.to_string())?;
        // Stamp the session for the history-rail colors BEFORE the rotation
        // (session_id is pre-rotation; the NewSession below clears it).
        s.mark_session_compacted(session_id).map_err(|e| e.to_string())?;
    }
    reload_context(&state).await?;

    // Rotate to a fresh session so the compacted history leaves the context
    // window; the old session remains in the DB for reference.
    {
        let tx = state.cmd_tx.lock().await;
        match tx.as_ref() {
            Some(sender) => sender
                .send(Command::NewSession)
                .await
                .map_err(|e| format!("Agent channel closed: {e}"))?,
            None => return Err("Not unlocked yet".into()),
        }
    }

    Ok(name)
}

/// Mode selector (Plan/Think/Auto) — apply the mode's approval-policy preset
/// live. The selector in the chatbox's send row calls this.
#[tauri::command]
pub async fn set_mode(state: State<'_, WebState>, mode: String) -> Result<(), String> {    let m = crate::config::Mode::parse(&mode)
        .ok_or_else(|| format!("Unknown mode '{mode}' (use plan, think, or auto)"))?;
    *state.mode.lock().await = m;
    let tx = state.cmd_tx.lock().await;
    match tx.as_ref() {
        Some(sender) => sender
            .send(Command::SetMode { mode: m })
            .await
            .map_err(|e| format!("Agent channel closed: {e}")),
        None => Err("Not unlocked yet".into()),
    }
}

/// Read the current reasoning mode (for the UI bar's active state).
#[tauri::command]
pub async fn get_mode(state: State<'_, WebState>) -> Result<String, String> {
    Ok(state.mode.lock().await.label().to_string())
}

/// Read the to-do list markdown (the shared plan shown in the chat's to-do
/// panel). Empty string when none exists.
#[tauri::command]
pub async fn get_todo(state: State<'_, WebState>) -> Result<String, String> {
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.get_setting("todo_markdown")
        .map_err(|e| e.to_string())
        .map(|v| v.unwrap_or_default())
}

/// Replace the to-do list markdown from the UI editor (Plan mode). Persists it
/// and pushes it into the runtime so the system prompt reflects the new plan.
#[tauri::command]
pub async fn set_todo(state: State<'_, WebState>, markdown: String) -> Result<(), String> {
    // 1. Persist (source of truth; the runtime does not persist on SetTodo).
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.set_setting("todo_markdown", &markdown)
            .map_err(|e| e.to_string())?;
    }
    // 2. Push into the runtime (updates Shared.todo + rebuilds the prompt).
    let tx = state.cmd_tx.lock().await;
    match tx.as_ref() {
        Some(sender) => sender
            .send(Command::SetTodo { markdown })
            .await
            .map_err(|e| format!("Agent channel closed: {e}")),
        None => Err("Not unlocked yet".into()),
    }
}

/// List all sub-agents (Panel 6).
#[tauri::command]
pub async fn list_sub_agents(state: State<'_, WebState>) -> Result<Vec<crate::db::store::SubAgent>, String> {
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.list_sub_agents().map_err(|e| e.to_string())
}

/// Create a new sub-agent. Returns its id.
#[tauri::command]
pub async fn create_sub_agent(
    state: State<'_, WebState>,
    name: String,
    description: String,
    persona: String,
    model: String,
) -> Result<i64, String> {
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.create_sub_agent(&name, &description, &persona, &model)
        .map_err(|e| e.to_string())
}

/// Update a sub-agent's editable fields.
#[tauri::command]
pub async fn update_sub_agent(
    state: State<'_, WebState>,
    id: i64,
    name: String,
    description: String,
    persona: String,
    model: String,
) -> Result<(), String> {
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.update_sub_agent(id, &name, &description, &persona, &model)
        .map_err(|e| e.to_string())
}

/// Delete a sub-agent.
#[tauri::command]
pub async fn delete_sub_agent(state: State<'_, WebState>, id: i64) -> Result<(), String> {
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.delete_sub_agent(id).map_err(|e| e.to_string())
}

/// Seven: should the post-login alpha Chronos pop-up be shown? True unless the
/// user dismissed it ("Don't show again").
#[tauri::command]
pub async fn should_show_alpha_popup(state: State<'_, WebState>) -> Result<bool, String> {
    let store = state.store.lock().await;
    if let Some(s) = store.as_ref() {
        let s = s.lock().await;
        let dismissed = s
            .get_setting("alpha_popup_dismissed")
            .map_err(|e| e.to_string())?;
        Ok(!dismissed.map(|v| v == "1").unwrap_or(false))
    } else {
        Ok(false)
    }
}

/// Seven: persist dismissal of the alpha Chronos pop-up ("Don't show again").
#[tauri::command]
pub async fn dismiss_alpha_popup(state: State<'_, WebState>) -> Result<(), String> {
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.set_setting("alpha_popup_dismissed", "1")
        .map_err(|e| e.to_string())
}

/// Approve a pending tool call.
#[tauri::command]
pub async fn approve(state: State<'_, WebState>, index: usize) -> Result<(), String> {
    let tx = state.cmd_tx.lock().await;
    match tx.as_ref() {
        Some(sender) => sender
            .send(Command::Approve { index })
            .await
            .map_err(|e| format!("Agent channel closed: {e}")),
        None => Err("Not unlocked yet".into()),
    }
}

/// Deny a pending tool call.
#[tauri::command]
pub async fn deny(state: State<'_, WebState>, index: usize) -> Result<(), String> {
    let tx = state.cmd_tx.lock().await;
    match tx.as_ref() {
        Some(sender) => sender
            .send(Command::Deny { index })
            .await
            .map_err(|e| format!("Agent channel closed: {e}")),
        None => Err("Not unlocked yet".into()),
    }
}

/// Start a fresh session.
#[tauri::command]
pub async fn new_session(state: State<'_, WebState>) -> Result<(), String> {
    let tx = state.cmd_tx.lock().await;
    match tx.as_ref() {
        Some(sender) => sender
            .send(Command::NewSession)
            .await
            .map_err(|e| format!("Agent channel closed: {e}")),
        None => Err("Not unlocked yet".into()),
    }
}

/// List past sessions (for a future resume UI).
#[tauri::command]
pub async fn list_sessions(state: State<'_, WebState>) -> Result<Vec<SessionSummary>, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.list_sessions().map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// One past session's messages for the history viewer (read-only rendering
/// in the chat). System rows are skipped — they're prompt plumbing, not
/// conversation.
#[tauri::command]
pub async fn load_session_messages(
    state: State<'_, WebState>,
    session_id: i64,
) -> Result<Vec<ChatMessage>, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            let msgs = s
                .load_messages(session_id, u32::MAX)
                .map_err(|e| e.to_string())?;
            Ok(msgs
                .into_iter()
                .filter(|m| m.role != ChatRole::System)
                .collect())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// List available Ollama models (for the model selector dropdown).
#[tauri::command]
pub async fn list_models(state: State<'_, WebState>) -> Result<Vec<String>, String> {
    state.provider.list_models().await.map_err(|e| e.to_string())
}

/// Which model backend is active: `"ollama"` or `"ambercore"`.
#[tauri::command]
pub async fn get_backend(state: State<'_, WebState>) -> Result<String, String> {
    Ok(state.config.lock().await.active_backend.clone())
}

/// Toggle the active model backend (`"ollama"` or `"ambercore"`). Persists to
/// config.toml and flips the live provider URL — no restart. The agent runtime
/// and health monitor pick up the new endpoint on their next call (they share
/// the same `Arc<dyn ModelProvider>` instance).
#[tauri::command]
pub async fn set_backend(
    state: State<'_, WebState>,
    backend: String,
) -> Result<(), String> {
    if backend != "ollama" && backend != "ambercore" {
        return Err(format!("unknown backend '{backend}' (expected ollama|ambercore)"));
    }
    // 1. Persist the choice (clearing any active cloud provider) + resolve URL.
    let new_url = {
        let mut cfg = state.config.lock().await;
        cfg.active_backend = backend.clone();
        cfg.active_provider_id = None;
        let cfg_clone = cfg.clone();
        drop(cfg);
        save_config(&state.paths, &cfg_clone).map_err(|e| format!("Save config: {e}"))?;
        cfg_clone.resolved_provider_url()
    };

    // 2. Switch the dispatch route to the local backend. This single shared
    //    instance is held (via Arc clones) by the agent runtime + health monitor,
    //    so the swap reaches both with no re-spawn.
    let lb = crate::model::dispatch::LocalBackend::parse(&backend)
        .unwrap_or(crate::model::dispatch::LocalBackend::Ollama);
    state.provider.set_local(lb, new_url).await;
    tracing::info!(%backend, "switched model backend");
    Ok(())
}

/// A `.gguf` file found on disk (read-only inventory — no inference wiring).
#[derive(Debug, Clone, Serialize)]
pub struct GgufFile {
    pub name: String,
    /// Human-readable size, e.g. "4.1 GB".
    pub size: String,
    pub path: String,
}

/// Whether a path has a `.gguf` extension (case-insensitive).
fn is_gguf_path(p: &std::path::Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.eq_ignore_ascii_case("gguf") || s.eq_ignore_ascii_case("safetensors"))
        .unwrap_or(false)
}

/// Collect `*.gguf` paths from a directory **and its immediate subfolders**
/// (the pull layout is `<model-name>/<model>.gguf`; flat files keep working).
/// Deeper nesting is not traversed — the models dir is user-owned, not a tree.
fn collect_gguf_paths(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    // Sharded safetensors checkpoints list only their first shard — the
    // engine stitches the rest via index.json.
    let listable = |p: &std::path::Path| -> bool {
        if !is_gguf_path(p) {
            return false;
        }
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else { return true };
        let lower = name.to_ascii_lowercase();
        if !lower.ends_with(".safetensors") {
            return true;
        }
        match lower.rfind("-of-") {
            None => true,
            Some(_) => lower.contains("-00001-of-"),
        }
    };
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if listable(&p) {
            out.push(p);
        } else if p.is_dir() {
            if let Ok(sub) = std::fs::read_dir(&p) {
                for e in sub.flatten() {
                    let sp = e.path();
                    if listable(&sp) {
                        out.push(sp);
                    }
                }
            }
        }
    }
    out
}

/// Scan a directory for `.gguf` model files (read-only inventory for the Models
/// panel). Returns an empty list if the directory is missing or unreadable.
#[tauri::command]
pub async fn scan_gguf_directory(dir: String) -> Result<Vec<GgufFile>, String> {
    let path = std::path::Path::new(&dir);
    if !path.is_dir() {
        return Err(format!("Not a directory: {dir}"));
    }
    let mut out = Vec::new();
    for p in collect_gguf_paths(path) {
        let name = p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(gguf)")
            .to_string();
        let size = match std::fs::metadata(&p) {
            Ok(m) => human_bytes(m.len()),
            Err(_) => "?".into(),
        };
        out.push(GgufFile {
            name,
            size,
            path: p.display().to_string(),
        });
    }
    // Largest first — usually what you want when browsing models.
    out.sort_by(|a, b| b.path.cmp(&a.path));
    Ok(out)
}

/// Get the persisted GGUF models directory (if the user set one).
#[tauri::command]
pub async fn get_gguf_directory(state: State<'_, WebState>) -> Result<Option<String>, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.get_setting("gguf_directory").map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// Persist the GGUF models directory so it's remembered across launches.
#[tauri::command]
pub async fn set_gguf_directory(state: State<'_, WebState>, dir: String) -> Result<(), String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.set_setting("gguf_directory", &dir).map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// Format a byte count as a short human-readable string.
fn human_bytes(n: u64) -> String {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    const KB: u64 = 1024;
    if n >= GB {
        format!("{:.1} GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.0} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.0} KB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}

// ---- Models panel v0.5: active route + AmberCore / Ollama / Provider API ----

/// Which backend is currently active, for highlighting the panel's box + row.
#[derive(Debug, Clone, Serialize)]
pub struct ActiveRouteInfo {
    /// `"local"` or `"cloud"`.
    pub kind: String,
    /// When local: `"ollama"` or `"ambercore"`. Absent for cloud.
    pub backend: Option<String>,
    /// When cloud: the active provider id. Absent for local.
    pub provider_id: Option<i64>,
    /// The currently active model name.
    pub model: String,
}

/// Read the active route + model so the panel can highlight the right box.
#[tauri::command]
pub async fn get_active_route(
    state: State<'_, WebState>,
) -> Result<ActiveRouteInfo, String> {
    Ok(active_route_info(&state).await)
}

/// Build the current route + model snapshot (shared by `get_active_route` and
/// the `model-changed` event payload).
async fn active_route_info(state: &WebState) -> ActiveRouteInfo {
    let cfg = state.config.lock().await;
    match state.provider.route().await {
        ActiveRoute::Local { backend } => ActiveRouteInfo {
            kind: "local".into(),
            backend: Some(backend.as_str().into()),
            provider_id: None,
            model: cfg.model.clone(),
        },
        ActiveRoute::Cloud { route } => ActiveRouteInfo {
            kind: "cloud".into(),
            backend: None,
            provider_id: Some(route.provider_id),
            model: cfg.model.clone(),
        },
    }
}

/// An AmberCore model row (blue box). Metadata is read from disk since AmberCore's
/// `/api/tags` only exposes the tag.
#[derive(Debug, Clone, Serialize)]
pub struct AmberCoreModel {
    pub name: String,
    pub quantization: String,
    pub downloaded_at: String,
    pub size: String,
}

/// Resolve the AmberCore models directory: the configured override, else the
/// Encapsulated default — a plain `models` folder BESIDE THE EXE (models are
/// public artifacts: they live on disk next to the app, never inside the
/// encrypted capsule, never in temp).
fn resolve_ambercore_dir(cfg: &crate::config::Config) -> Option<std::path::PathBuf> {
    if let Some(d) = cfg.ambercore_models_dir_path() {
        return Some(d);
    }
    Some(crate::config::encaps_models_dir(
        &crate::config::Paths::default_data_dir(),
    ))
}

/// Parse a quantization tag out of a GGUF filename, e.g.
/// `Qwen2-0.5B-Instruct-Q4_K_M.gguf` → `Q4_K_M`. Returns `"-"` when unknown.
fn parse_quant(filename: &str) -> String {
    let stem = filename
        .trim_end_matches(".gguf")
        .trim_end_matches(".GGUF");
    let upper = stem.to_ascii_uppercase();
    // Common GGUF quant labels. Match the longest first for correctness.
    let labels = [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0",
        "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32", "BF16",
    ];
    for label in labels {
        if upper.contains(label) {
            return label.to_string();
        }
    }
    "—".to_string()
}

/// List AmberCore models by scanning the models directory. Falls back to the live
/// server's `/api/tags` (tags only, no metadata) when a server is running but no
/// directory is configured.
#[tauri::command]
pub async fn list_ambercore_models(
    state: State<'_, WebState>,
) -> Result<Vec<AmberCoreModel>, String> {
    let cfg = state.config.lock().await.clone();

    // Remote mode: query the remote server's /api/tags instead of the local disk.
    if cfg.ambercore_remote {
        let url = format!("{}/api/tags", cfg.ambercore_url.trim_end_matches('/'));
        let client = reqwest::Client::builder().build().map_err(|e| e.to_string())?;
        let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("Remote AmberCore {url} returned {}", resp.status()));
        }
        #[derive(serde::Deserialize)]
        struct TagsResp {
            #[serde(default)]
            models: Vec<TagsModel>,
        }
        #[derive(serde::Deserialize)]
        struct TagsModel {
            name: String,
        }
        let parsed: TagsResp = resp.json().await.map_err(|e| e.to_string())?;
        return Ok(parsed
            .models
            .into_iter()
            .map(|m| AmberCoreModel {
                name: m.name,
                quantization: "—".into(),
                downloaded_at: "—".into(),
                size: "—".into(),
            })
            .collect());
    }

    let dir = resolve_ambercore_dir(&cfg);
    let mut out = Vec::new();

    // 1. Scan the configured/native directory (rich metadata) — flat files
    //    plus one level of per-model subfolders (the pull layout).
    if let Some(dir) = dir {
        if dir.is_dir() {
            for path in collect_gguf_paths(&dir) {
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("model")
                    .to_string();
                let meta = std::fs::metadata(&path).ok();
                let size = meta.as_ref().map(|m| human_bytes(m.len())).unwrap_or_default();
                let downloaded_at = meta
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| {
                        t.duration_since(std::time::UNIX_EPOCH)
                            .ok()
                            .map(|d| {
                                chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                                    .map(|dt| dt.format("%Y-%m-%d").to_string())
                            })
                    })
                    .flatten()
                    .unwrap_or_else(|| "—".to_string());
                out.push(AmberCoreModel {
                    name: filename.trim_end_matches(".gguf").to_string(),
                    quantization: parse_quant(&filename),
                    downloaded_at,
                    size,
                });
            }
        }
    }

    // 2. No directory found / empty — try the live server (tags only).
    if out.is_empty() {
        if let Ok(tags) = state.provider.list_models().await {
            for name in tags {
                out.push(AmberCoreModel {
                    name,
                    quantization: "—".into(),
                    downloaded_at: "—".into(),
                    size: "—".into(),
                });
            }
        }
    }

    out.sort_by_key(|m| m.name.to_ascii_lowercase());
    Ok(out)
}

/// Persist (or clear) the AmberCore custom models directory.
#[tauri::command]
pub async fn set_ambercore_directory(
    state: State<'_, WebState>,
    dir: Option<String>,
) -> Result<(), String> {
    let mut cfg = state.config.lock().await;
    cfg.ambercore_models_dir = dir.filter(|s| !s.trim().is_empty());
    let cfg_clone = cfg.clone();
    drop(cfg);
    save_config(&state.paths, &cfg_clone).map_err(|e| format!("Save config: {e}"))?;
    // Point the embedded engine's catalog at the (new) directory so pulls,
    // listings, and generation all see it.
    if let Some(d) = resolve_ambercore_dir(&cfg_clone) {
        if let Err(e) = state.provider.embedded().reload_catalog(d).await {
            tracing::warn!("embedded AmberCore catalog reload failed: {e}");
        }
    }
    Ok(())
}

/// Read the AmberCore custom models directory, if set.
#[tauri::command]
pub async fn get_ambercore_directory(state: State<'_, WebState>) -> Result<Option<String>, String> {
    Ok(state.config.lock().await.ambercore_models_dir.clone())
}

/// Delete an AmberCore model (the models panel's bin button): unload any
/// pooled replica (releasing the GGUF mmap so Windows allows the delete),
/// drop every catalog spelling of the tag (persisting to `manifest.json`),
/// then remove the GGUF, its sibling `<stem>.tokenizer.json`, and — when the
/// model lives in its own per-model pull folder with no other GGUFs — the
/// folder itself. Refused in remote-server mode (no such API over HTTP).
#[tauri::command]
pub async fn delete_ambercore_model(
    state: State<'_, WebState>,
    name: String,
) -> Result<(), String> {
    let cfg = state.config.lock().await.clone();
    if cfg.ambercore_remote {
        return Err(
            "Remote AmberCore servers can't be deleted from Phoenix — remove \
             the model on the server itself."
                .into(),
        );
    }
    let Some(models_dir) = resolve_ambercore_dir(&cfg) else {
        return Err("No AmberCore models directory is configured.".into());
    };
    // The panel lists file stems, so match the GGUF by stem.
    let gguf = collect_gguf_paths(&models_dir)
        .into_iter()
        .find(|p| p.file_stem().and_then(|s| s.to_str()) == Some(name.as_str()))
        .ok_or_else(|| format!("No GGUF found for '{name}' in {}", models_dir.display()))?;

    // 1. Engine side first: unload replicas + catalog removal (manifest).
    state
        .provider
        .embedded()
        .remove_model(&name)
        .await
        .map_err(|e| format!("AmberCore unload/remove: {e}"))?;

    // 2. File deletion (blocking thread — GGUFs are GBs).
    let moved_gguf = gguf.clone();
    let moved_root = models_dir.clone();
    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        std::fs::remove_file(&moved_gguf)?;
        let folder = moved_gguf.parent().unwrap_or(std::path::Path::new("."));
        let stem = moved_gguf
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default();
        // Model-specific tokenizer (the pull layout always writes this name).
        let _ = std::fs::remove_file(folder.join(format!("{stem}.tokenizer.json")));
        // A shared generic tokenizer.json is only safe to remove when it sits
        // inside the model's OWN folder (per-model pull layout) — flat models
        // may share it with siblings. Removing that folder removes it too.
        let is_own_folder = folder != moved_root;
        if is_own_folder
            && std::fs::read_dir(folder)
                .map(|rd| {
                    rd.flatten()
                        .all(|e| !is_gguf_path(&e.path()))
                })
                .unwrap_or(false)
        {
            let _ = std::fs::remove_dir_all(folder);
        }
        Ok(())
    })
    .await
    .map_err(|e| format!("delete task join: {e}"))?
    .map_err(|e| format!("delete files: {e}"))?;

    // 3. Refresh the engine's catalog so the tag disappears everywhere.
    if let Err(e) = state.provider.embedded().reload_catalog(models_dir).await {
        tracing::warn!("embedded AmberCore catalog reload after delete failed: {e}");
    }
    Ok(())
}

// ---- Model search (the AmberCore "find models" modal) ----------------------

/// One downloadable model file, as shown on a search card.
#[derive(Serialize)]
pub struct ModelSearchResult {
    /// Display name (repo / model name — the file's quant is separate).
    pub name: String,
    /// The GGUF (or downloadable) file name.
    pub file_name: String,
    /// File size in GiB (rounded) — drives the fit coloring with the ×1.5 + 300 MiB
    /// resident estimate the engine's pre-flight check uses.
    pub size_gb: f64,
    /// Parameter count as displayed (e.g. "8B"), parsed from the names.
    pub params: Option<String>,
    /// Quantization factor (e.g. "Q4_K_M"), parsed from the file name.
    pub quant: Option<String>,
    /// Direct download URL (feeds `pull_ambercore_model`).
    pub url: String,
    /// "huggingface" | "civitai".
    pub source: String,
    /// Civitai hosts image-gen models (Flux/SD checkpoints + LoRAs) — they are
    /// GGUFs sometimes, but NOT text-model GGUFs AmberCore can run. The card
    /// shows them with a badge and Pull disabled.
    pub ambercore_compatible: bool,
}

/// Search response: results + live VRAM (MiB) — the TOTAL powers the modal's
/// "GPU: … GB" line, the FREE amount drives the fit coloring (fits → green,
/// ≤ 1 GiB spill → orange) with the same estimate the engine's pre-flight
/// refusal uses.
#[derive(Serialize)]
pub struct ModelSearchResponse {
    pub vram_total_mb: Option<u64>,
    pub vram_free_mb: Option<u64>,
    pub results: Vec<ModelSearchResult>,
}

/// Search Hugging Face (GGUF LLMs — sizes via the repo tree API, fetched
/// concurrently) or Civitai (image-gen models, browsable but not
/// AmberCore-runnable). An empty query lists the most-downloaded GGUFs.
#[tauri::command]
pub async fn search_models(
    query: String,
    source: String,
) -> Result<ModelSearchResponse, String> {
    let client = reqwest::Client::builder()
        .user_agent("Phoenix-Agent/0.8 model-search")
        .build()
        .map_err(|e| format!("build client: {e}"))?;
    let mut results = match source.as_str() {
        "civitai" => search_civitai(&client, &query).await?,
        _ => search_huggingface(&client, &query).await?,
    };
    let (vram_total_mb, vram_free_mb) =
        ambercore::backend::query_vram_mb().map(|(t, f)| (Some(t), Some(f))).unwrap_or((None, None));
    // Relevance tuning: the modal is for models the user's GPU can REALISTICALLY
    // run — drop anything that would spill more than 1 GiB (a small spill is a
    // deliberate concession; a big one is the WDDM crawl). Uses the same
    // ×1.5 + 300 MiB resident estimate as the pre-flight refusal. Unknown
    // sizes (blocked size lookups) are kept — the card shows no tint.
    const MAX_SPILL_MB: u64 = 1024;
    if let Some(free) = vram_free_mb {
        results.retain(|m| {
            let size_mb = (m.size_gb * 1024.0).round() as u64;
            size_mb == 0 || size_mb * 3 / 2 + 300 <= free + MAX_SPILL_MB
        });
    }
    Ok(ModelSearchResponse {
        vram_total_mb,
        vram_free_mb,
        results,
    })
}

/// Hugging Face: `/api/models?filter=gguf` (sorted by downloads), then per-repo
/// tree lookups for the GGUF file sizes — spawned concurrently, top repos only.
async fn search_huggingface(
    client: &reqwest::Client,
    query: &str,
) -> Result<Vec<ModelSearchResult>, String> {
    let mut url =
        reqwest::Url::parse("https://huggingface.co/api/models").map_err(|e| e.to_string())?;
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("filter", "gguf");
        q.append_pair("sort", "downloads");
        q.append_pair("direction", "-1");
        q.append_pair("limit", "12");
        q.append_pair("expand[]", "siblings");
        if !query.trim().is_empty() {
            q.append_pair("search", query.trim());
        }
    }
    let resp: serde_json::Value = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Hugging Face search request failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("Hugging Face search failed: {e}"))?
        .json()
        .await
        .map_err(|e| format!("Hugging Face search parse failed: {e}"))?;

    // Repo -> its GGUF file names (single-file only — shards are rejected at
    // pull time; prefer the common quants, cap 3 per repo).
    let mut repos: Vec<(String, Vec<String>)> = Vec::new();
    for item in resp.as_array().into_iter().flatten() {
        let id = item["id"].as_str().unwrap_or_default().to_string();
        if id.is_empty() {
            continue;
        }
        let mut files: Vec<String> = item["siblings"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| s["rfilename"].as_str())
            .filter(|f| f.to_ascii_lowercase().ends_with(".gguf") && !is_shard_gguf(f))
            .map(str::to_string)
            .collect();
        files.sort_by_key(|f| quant_preference(f));
        files.truncate(3);
        if !files.is_empty() {
            repos.push((id, files));
        }
    }

    // Concurrent size lookups: one task per repo resolves its GGUF sizes
    // (best-effort — Hugging Face blocks the tree endpoint from some
    // networks/VPNs; unknown sizes render as "—" cards with no fit tint).
    let mut handles = Vec::new();
    for (repo, files) in repos {
        let client = client.clone();
        handles.push(tokio::spawn(async move {
            let sizes = fetch_hf_gguf_sizes(&client, &repo, &files).await;
            (repo, files, sizes)
        }));
    }
    let mut out = Vec::new();
    for handle in handles {
        let (repo, files, sizes) = handle
            .await
            .map_err(|e| format!("size lookup task failed: {e}"))?;
        for file in files {
            let size_bytes = sizes.get(&file).copied().unwrap_or(0);
            let display = file
                .rsplit('/')
                .next()
                .unwrap_or(&file)
                .trim_end_matches(".gguf")
                .to_string();
            out.push(ModelSearchResult {
                name: repo.clone(),
                file_name: display.clone(),
                size_gb: (size_bytes as f64 / 1_073_741_824.0 * 10.0).round() / 10.0,
                params: parse_params(&format!("{repo}/{file}")),
                quant: {
                    let q = parse_quant(&file);
                    (q != "—").then_some(q)
                },
                url: format!("https://huggingface.co/{repo}/resolve/main/{file}"),
                source: "huggingface".into(),
                ambercore_compatible: true,
            });
        }
    }
    out.sort_by(|a, b| b.size_gb.partial_cmp(&a.size_gb).unwrap_or(std::cmp::Ordering::Equal));
    out.truncate(36);
    Ok(out)
}

/// `…-00001-of-00002.gguf` — split/sharded GGUF. AmberCore loads single-file
/// GGUFs only (rejected at pull time), so search never offers them.
fn is_shard_gguf(name: &str) -> bool {
    let stem = name.trim_end_matches(".gguf");
    stem.rsplit_once("-of-")
        .map(|(_, tail)| tail.len() == 5 && tail.chars().all(|c| c.is_ascii_digit()))
        .unwrap_or(false)
}

/// Smaller = more likely what a user wants to see first (the common quants).
fn quant_preference(file: &str) -> u8 {
    let f = file.to_ascii_uppercase();
    for (i, label) in [
        "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0", "Q4_0", "Q3_K_M", "IQ4",
    ]
    .iter()
    .enumerate()
    {
        if f.contains(label) {
            return i as u8;
        }
    }
    9
}

/// Fetch the sizes of the given GGUF files from a HF repo's tree.
async fn fetch_hf_gguf_sizes(
    client: &reqwest::Client,
    repo: &str,
    files: &[String],
) -> std::collections::HashMap<String, u64> {
    let mut sizes = std::collections::HashMap::new();
    let Ok(resp) = client
        .get(format!("https://huggingface.co/api/models/{repo}/tree/main"))
        .send()
        .await
    else {
        return sizes;
    };
    let Ok(tree) = resp.json::<serde_json::Value>().await else {
        return sizes;
    };
    for entry in tree.as_array().into_iter().flatten() {
        let path = entry["path"].as_str().unwrap_or_default().to_string();
        if files.iter().any(|f| *f == path) {
            if let Some(size) = entry["size"].as_u64() {
                sizes.insert(path, size);
            }
        }
    }
    sizes
}

/// Civitai: image-gen models (checkpoints/LoRAs — browsable, but NOT
/// AmberCore-runnable; cards carry the badge + Pull stays disabled).
async fn search_civitai(
    client: &reqwest::Client,
    query: &str,
) -> Result<Vec<ModelSearchResult>, String> {
    let mut url =
        reqwest::Url::parse("https://civitai.com/api/v1/models").map_err(|e| e.to_string())?;
    {
        let mut q = url.query_pairs_mut();
        q.append_pair("limit", "24");
        if !query.trim().is_empty() {
            q.append_pair("query", query.trim());
        }
    }
    let resp: serde_json::Value = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Civitai search request failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("Civitai search failed: {e}"))?
        .json()
        .await
        .map_err(|e| format!("Civitai search parse failed: {e}"))?;

    let mut out = Vec::new();
    for item in resp["items"].as_array().into_iter().flatten() {
        let name = item["name"].as_str().unwrap_or_default().to_string();
        if name.is_empty() {
            continue;
        }
        // Newest version's files (the API returns newest first).
        let Some(version) = item["modelVersions"].as_array().and_then(|v| v.first()) else {
            continue;
        };
        for file in version["files"].as_array().into_iter().flatten().take(2) {
            let Some(download) = file["downloadUrl"].as_str() else { continue };
            let file_name = file["name"].as_str().unwrap_or(&name).to_string();
            let size_kb = file["sizeKB"].as_f64().unwrap_or(0.0);
            if size_kb <= 0.0 {
                continue;
            }
            out.push(ModelSearchResult {
                name: name.clone(),
                file_name: file_name.clone(),
                size_gb: (size_kb / 1_048_576.0 * 10.0).round() / 10.0,
                params: parse_params(&format!("{name} {file_name}")),
                quant: {
                    let q = parse_quant(&file_name);
                    (q != "—").then_some(q)
                },
                url: download.to_string(),
                source: "civitai".into(),
                ambercore_compatible: false,
            });
        }
    }
    out.truncate(36);
    Ok(out)
}

/// Pull a parameter count like "8B" / "0.5B" / "14B" out of model/file names.
fn parse_params(text: &str) -> Option<String> {
    let chars: Vec<char> = text.chars().collect();
    for i in 0..chars.len() {
        if chars[i].is_ascii_digit() {
            let mut j = i;
            while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '.') {
                j += 1;
            }
            // "8B", "0.5B" — a unit B right after the number (case-insensitive).
            if j < chars.len() && (chars[j] == 'B' || chars[j] == 'b') {
                // Avoid matching inside words like "Q4B" quants or hex-ish runs.
                let prev_ok = i == 0 || !chars[i - 1].is_ascii_alphanumeric();
                let num: String = chars[i..j].iter().collect();
                if prev_ok && num.parse::<f64>().map(|n| n > 0.0 && n < 2000.0).unwrap_or(false) {
                    return Some(format!("{num}B"));
                }
            }
        }
    }
    None
}

/// Download a GGUF model **and its tokenizer** from a URL into the AmberCore
/// models directory, then register it with AmberCore. The tokenizer is fetched
/// automatically for Hugging Face URLs (same repo, then the base model repo);
/// `tokenizer_url` overrides the derivation for other sources. A pull is only
/// complete when both files exist — without a tokenizer AmberCore cannot load
/// the model. Streams progress to the frontend via the
/// `ambercore-pull-progress` Tauri event (`phase`: `model` | `tokenizer`).
/// Returns the derived tag.
#[tauri::command]
pub async fn pull_ambercore_model(
    app: AppHandle,
    state: State<'_, WebState>,
    url: String,
    tokenizer_url: Option<String>,
) -> Result<String, String> {
    use tauri::Emitter;
    let cfg = state.config.lock().await.clone();
    let dir = resolve_ambercore_dir(&cfg).ok_or_else(|| {
        "No AmberCore models directory configured and home dir unknown".to_string()
    })?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("create models dir: {e}"))?;

    // Direct-download URL: normalize HF page links (blob → resolve). Non-HF
    // URLs are used verbatim — they may carry query strings that matter.
    let model_url = super::model_urls::hf_file_url(&url).unwrap_or_else(|| url.clone());
    // Filename from the URL's last path segment, query string stripped.
    let filename = super::model_urls::filename_from_url(&url)
        .unwrap_or_else(|| format!("model-{}.gguf", chrono::Utc::now().timestamp()));

    // Split/sharded GGUFs (e.g. `-00001-of-00003.gguf`) can never load in
    // AmberCore (single-file loader) — fail BEFORE the multi-GB download.
    // Sharded SAFETENSORS repos are fine: the engine's multi-region reader
    // stitches them via the repo's index.json.
    let is_safetensors = filename.to_ascii_lowercase().ends_with(".safetensors");
    if super::model_urls::is_split_gguf(&filename) && !is_safetensors {
        return Err(format!(
            "`{filename}` is one shard of a split GGUF. AmberCore loads single-file              GGUFs only — pick a quant that is one file (no `-00001-of-000NN`              shards) and Pull again."
        ));
    }

    // Every progress event carries the pull's id (the file stem) so several
    // concurrent pulls each drive their own bar in the Models panel; a final
    // done/error event retires the bar without relying on the invoke result.
    let pull_id = filename
        .trim_end_matches(".gguf")
        .trim_end_matches(".GGUF")
        .to_string();
    let app_progress = app.clone();
    let emit = |phase: &'static str, completed: u64, total: Option<u64>| {
        let payload = serde_json::json!({
            "id": pull_id,
            "phase": phase,
            "completed": completed,
            "total": total,
        });
        let _ = app_progress.emit("ambercore-pull-progress", payload);
    };

    // Sharded-safetensors metadata: folder/tag from the checkpoint base stem
    // + repo name, and the repo base for index.json / sibling fetches.
    let shard_info: Option<(String, Option<String>, String)> =
        if is_safetensors && super::model_urls::is_split_gguf(&filename) {
            let base_stem = super::model_urls::shard_base_stem(
                filename.trim_end_matches(".safetensors"),
            )
            .unwrap_or_else(|| "model".into());
            let repo_base = model_url
                .split("/resolve/")
                .next()
                .map(str::to_string)
                .unwrap_or_else(|| model_url.clone());
            let repo_name = repo_base
                .rsplit('/')
                .next()
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            Some((base_stem, repo_name, repo_base))
        } else {
            None
        };

    // The whole flow runs inside one block so every exit path (download
    // failure, unsupported arch, missing tokenizer, register error) can retire
    // the pull's progress bar with a terminal done/error event.
    let outcome: Result<String, String> = async {
        if let Some((base_stem, repo_name, repo_base)) = &shard_info {
            return super::shard_pull::pull_sharded_safetensors(
                &state,
                &dir,
                &model_url,
                &filename,
                base_stem,
                repo_name.clone(),
                repo_base,
                &pull_id,
                &app,
            )
            .await;
        }
        // Per-model folder: `<models_dir>/<model-name>/` holds the GGUF + its
        // tokenizer (and any future per-model files) so models can never pick up
        // each other's tokenizer. Flat files already on disk keep working.
        let folder = super::model_urls::model_folder_name(&filename);
        let model_dir = dir.join(&folder);
        std::fs::create_dir_all(&model_dir)
            .map_err(|e| format!("create model folder {}: {e}", model_dir.display()))?;
        let dest = model_dir.join(&filename);

        let client = reqwest::Client::builder()
            .build()
            .map_err(|e| format!("build client: {e}"))?;

        // 1. Model — skip the (multi-GB) download when the file is already on
        //    disk, so a retry after a failed tokenizer fetch is instant.
        let model_present = dest.is_file()
            && std::fs::metadata(&dest).map(|m| m.len() > 0).unwrap_or(false);
        if model_present {
            tracing::info!(?dest, "AmberCore pull: model already on disk, skipping download");
        } else {
            download_to_file(&client, &model_url, &dest, "model", &emit).await?;
        }

        // 1b. F16 safetensors models load through the content shim, which
        //     needs the repo's config.json next to the weights.
        let is_st = filename
            .to_ascii_lowercase()
            .ends_with(".safetensors");
        if is_st {
            let cfg_dest = model_dir.join("config.json");
            if !cfg_dest.is_file() {
                let repo = model_url
                    .rsplit_once('/')
                    .and_then(|(base, _)| base.rsplit_once('/'))
                    .map(|(root, repo)| format!("{root}/{repo}"))
                    .unwrap_or_else(|| model_url.clone());
                let cfg_url = format!("{}/resolve/main/config.json", repo.trim_end_matches('/'));
                if let Err(e) = download_to_file(&client, &cfg_url, &cfg_dest, "config", &emit).await {
                    tracing::warn!("config.json download failed ({e}) — F16 load will fail without it");
                }
            }
        }

        // 2. Architecture check — read the GGUF header (or the safetensors
        //    config.json) and reject architectures AmberCore can't run, BEFORE
        //    spending time on the tokenizer or registering an unloadable
        //    model. (Learned the hard way: a Qwen3.5 hybrid `qwen35` GGUF
        //    pulls fine and then dies at load time deep inside the qwen3
        //    builder with a missing-tensor error.)
        let arch = ambercore::model::gguf::probe_arch(&dest)
            .map_err(|e| format!("downloaded file is not a loadable GGUF: {e}"))?;
        if !ambercore::model::registry::is_supported(&arch) {
            return Err(format!(
                "AmberCore can't run the `{arch}` architecture yet (supported: {}). \
                 The model was kept at {} — it will work here if support is added later.",
                ambercore::model::registry::SUPPORTED_ARCHS.join(", "),
                dest.display(),
            ));
        }

        // 3. Tokenizer — AmberCore looks for `<stem>.tokenizer.json` (or a
        //    generic `tokenizer.json`) next to the GGUF, i.e. inside the model's
        //    own folder. The model-specific name lets multiple quants of the same
        //    model share the folder without collisions.
        let stem = dest
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let tok_dest = model_dir.join(format!("{stem}.tokenizer.json"));
        let have_tokenizer = tok_dest.is_file() || model_dir.join("tokenizer.json").is_file();
        if !have_tokenizer {
            // Explicit URL first, then whatever the model URL implies.
            let mut candidates: Vec<String> = Vec::new();
            if let Some(explicit) = tokenizer_url
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                candidates.push(explicit.to_string());
            }
            candidates.extend(super::model_urls::tokenizer_candidates(&model_url));

            let mut failures: Vec<String> = Vec::new();
            for cand in &candidates {
                match download_to_file(&client, cand, &tok_dest, "tokenizer", &emit).await {
                    Ok(()) => {
                        tracing::info!(cand, "AmberCore pull: tokenizer downloaded");
                        break;
                    }
                    Err(e) => {
                        // Drop any partial file so the next candidate starts clean.
                        let _ = tokio::fs::remove_file(&tok_dest).await;
                        failures.push(format!("{cand} → {e}"));
                    }
                }
            }
            if !tok_dest.is_file() {
                return Err(format!(
                    "model downloaded to {}, but no tokenizer was found. Tried:\n  {}\n\
                     Paste a direct tokenizer.json URL in the Tokenizer field and Pull \
                     again (the model will not re-download), or place a \
                     `{stem}.tokenizer.json` next to the GGUF in {}.",
                    dest.display(),
                    if failures.is_empty() {
                        "nothing — the URL is not a Hugging Face link and no \
                         tokenizer URL was given"
                            .to_string()
                    } else {
                        failures.join("\n  ")
                    },
                    model_dir.display(),
                ));
            }
        }

        // Register the model with the embedded engine's catalog — in-process
        // (persists to the dir's manifest.json); no `ambercore` binary involved.
        // The manifest `file` is folder-relative (resolved against the models dir).
        let tag = derive_ambercore_tag(&filename);
        let rel_path = format!("{folder}/{filename}");
        state
            .provider
            .embedded()
            .register_model(&tag, &rel_path)
            .await
            .map_err(|e| format!("register model: {e}"))?;
        tracing::info!(%tag, path = %rel_path, "registered AmberCore model (embedded catalog)");
        Ok(tag)
    }
    .await;
    {
        use tauri::Emitter;
        let payload = match &outcome {
            Ok(tag) => serde_json::json!({ "id": pull_id, "phase": "done", "tag": tag }),
            Err(e) => serde_json::json!({ "id": pull_id, "phase": "error", "error": e }),
        };
        let _ = app.emit("ambercore-pull-progress", payload);
    }
    outcome
}

/// Stream a URL to a file, emitting `phase`-tagged pull-progress events.
/// Fails on non-success HTTP status before touching the destination file.
pub(super) async fn download_to_file<F>(
    client: &reqwest::Client,
    url: &str,
    dest: &std::path::Path,
    phase: &'static str,
    emit: &F,
) -> Result<(), String>
where
    F: Fn(&'static str, u64, Option<u64>),
{
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }
    let total = resp.content_length();
    use futures::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::File::create(dest)
        .await
        .map_err(|e| format!("create file: {e}"))?;
    use tokio::io::AsyncWriteExt;
    let mut completed: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("read chunk: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("write chunk: {e}"))?;
        completed += chunk.len() as u64;
        emit(phase, completed, total);
    }
    file.flush().await.map_err(|e| format!("flush: {e}"))?;
    Ok(())
}

/// Derive an Ollama-style tag from a GGUF filename.
fn derive_ambercore_tag(filename: &str) -> String {
    let stem = filename
        .trim_end_matches(".gguf")
        .trim_end_matches(".GGUF")
        .trim_end_matches(".safetensors");
    // If the stem already looks like `name-Nunit` (e.g. `qwen2-7b`), use it; else
    // append `:latest`.
    if stem.contains('-') {
        stem.to_string()
    } else {
        format!("{stem}:latest")
    }
}

/// Start the AmberCore server (stopping any other backend), switch to it, and set
/// the active model. The "Run" semantics for the AmberCore box.
#[tauri::command]
pub async fn run_ambercore(
    app: AppHandle,
    state: State<'_, WebState>,
    model_tag: String,
) -> Result<(), String> {
    let cfg = state.config.lock().await.clone();
    let remote = cfg.ambercore_remote;
    // 1. Ensure the right engine is active.
    //    Remote mode: stop any local backend; Phoenix talks to the URL over HTTP.
    //    Local mode: the **embedded in-process engine** — nothing to spawn.
    state.process_mgr.stop_all().await;
    // 2. Persist + switch the route to AmberCore.
    if remote {
        let local_url = {
            let mut c = state.config.lock().await;
            c.active_backend = "ambercore".into();
            c.active_provider_id = None;
            let clone = c.clone();
            drop(c);
            save_config(&state.paths, &clone).map_err(|e| format!("Save config: {e}"))?;
            clone.ambercore_url.clone()
        };
        state.provider.set_local(LocalBackend::AmberCore, local_url).await;
    } else {
        {
            let mut c = state.config.lock().await;
            c.active_backend = "ambercore".into();
            c.active_provider_id = None;
            let clone = c.clone();
            drop(c);
            save_config(&state.paths, &clone).map_err(|e| format!("Save config: {e}"))?;
        }
        state.provider.set_local_embedded().await;
    }
    // 3. Switch the active model.
    apply_model(&app, &state, model_tag.clone()).await?;
    // 4. Warm the model into the embedded engine's pool (background — loads the
    //    GGUF so the first message doesn't pay the cold-start). No-op remotely.
    if !remote {
        let provider = state.provider.clone();
        let app = app.clone();
        tokio::spawn(async move {
            provider.warm_ambercore_model(&model_tag).await;
            emit_vram_status(&app, &provider);
        });
    }
    Ok(())
}

/// Surface the engine's VRAM verdict as a chat status message — either the
/// pre-flight REFUSAL (model would spill; loading was blocked) or the
/// "contained in VRAM" confirmation. No-op when the check couldn't run.
fn emit_vram_status(app: &AppHandle, provider: &std::sync::Arc<crate::model::dispatch::DispatchProvider>) {
    let state = provider.embedded().state();
    let message = state
        .last_vram_warning()
        .or_else(|| state.last_vram_info());
    if let Some(message) = message {
        let _ = app.emit(
            "agent-event",
            serde_json::json!({ "type": "status", "message": message }),
        );
    }
}

/// Link a **remote** AmberCore server (e.g. an AmberCore-Server installed on a
/// private machine). Persists the URL + remote flag, stops any local backend,
/// and points the model provider at the remote URL. Does not select a model —
/// the caller refreshes the list and the user clicks Run on one.
#[tauri::command]
pub async fn connect_ambercore_remote(
    state: State<'_, WebState>,
    url: String,
) -> Result<(), String> {
    let url = url.trim().trim_end_matches('/').to_string();
    if url.is_empty() {
        return Err("Remote server URL is empty.".into());
    }
    state.process_mgr.stop_all().await;
    {
        let mut c = state.config.lock().await;
        c.ambercore_url = url.clone();
        c.ambercore_remote = true;
        c.active_backend = "ambercore".into();
        c.active_provider_id = None;
        let clone = c.clone();
        drop(c);
        save_config(&state.paths, &clone).map_err(|e| format!("Save config: {e}"))?;
    }
    state.provider.set_local(LocalBackend::AmberCore, url).await;
    Ok(())
}

/// Switch AmberCore back to **local** mode (the embedded in-process engine).
#[tauri::command]
pub async fn use_local_ambercore(state: State<'_, WebState>) -> Result<(), String> {
    {
        let mut c = state.config.lock().await;
        c.ambercore_remote = false;
        c.active_backend = "ambercore".into();
        c.active_provider_id = None;
        let clone = c.clone();
        drop(c);
        save_config(&state.paths, &clone).map_err(|e| format!("Save config: {e}"))?;
    }
    state.provider.set_local_embedded().await;
    Ok(())
}

/// The current AmberCore connection mode + URL (for the Models panel UI).
#[derive(Debug, Clone, Serialize)]
pub struct AmberCoreStatus {
    pub remote: bool,
    pub url: String,
}

#[tauri::command]
pub async fn get_ambercore_status(state: State<'_, WebState>) -> Result<AmberCoreStatus, String> {
    let c = state.config.lock().await;
    Ok(AmberCoreStatus {
        remote: c.ambercore_remote,
        url: c.ambercore_url.clone(),
    })
}

/// Hardware check-up for the Telemetry tab (Main Menu → Telemetry). The CPU /
/// cores / RAM / OS baseline is captured once at engine boot (= app launch);
/// the GPU reading (name + live VRAM) refreshes on every call. CPU-only
/// builds report `backend: "cpu"` and no GPU.
#[tauri::command]
pub async fn get_hardware_status(
    state: State<'_, WebState>,
) -> Result<ambercore::server::telemetry::HardwareStatus, String> {
    Ok(state.provider.embedded().state().hardware_status())
}

/// An Ollama model row (yellow box). Real Ollama exposes `modified_at`; AmberCore
/// (when probed as an Ollama-compatible fallback) does not.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaModel {
    pub name: String,
    pub downloaded_at: String,
}

/// List Ollama models from the running server's `/api/tags`.
#[tauri::command]
pub async fn list_ollama_models(state: State<'_, WebState>) -> Result<Vec<OllamaModel>, String> {
    // Query the Ollama server directly (not the dispatch route, which may point
    // elsewhere). Ollama's tags response includes modified_at + details.
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| e.to_string())?;
    let cfg = state.config.lock().await;
    let url = format!("{}/api/tags", cfg.ollama_url.trim_end_matches('/'));
    drop(cfg);
    let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("GET {url} returned {}", resp.status()));
    }
    #[derive(serde::Deserialize)]
    struct TagsResp {
        #[serde(default)]
        models: Vec<TagsModel>,
    }
    #[derive(serde::Deserialize)]
    struct TagsModel {
        name: String,
        #[serde(default)]
        modified_at: Option<String>,
    }
    let parsed: TagsResp = resp.json().await.map_err(|e| e.to_string())?;
    let out = parsed
        .models
        .into_iter()
        .map(|m| OllamaModel {
            name: m.name,
            downloaded_at: m
                .modified_at
                .and_then(|s| s.get(..10).map(|x| x.to_string()))
                .unwrap_or_else(|| "—".to_string()),
        })
        .collect();
    Ok(out)
}

/// One-off reachability probe for an Ollama-compatible backend: `GET /api/tags`
/// with a short timeout. Used by boot-time auto-detection ONLY — the recurring
/// health monitor goes through the `DispatchProvider` (active route) instead, so
/// this never touches an inactive port during normal operation.
async fn endpoint_reachable(client: &reqwest::Client, base_url: &str) -> bool {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    match client.get(&url).send().await {
        Ok(r) => r.status().is_success(),
        Err(_) => false,
    }
}

/// Pull an Ollama-hosted model via `ollama pull`, streaming progress to the
/// frontend via the `ollama-pull-progress` event. Returns the model name.
#[tauri::command]
pub async fn pull_ollama_model(
    app: AppHandle,
    state: State<'_, WebState>,
    name: String,
) -> Result<String, String> {
    use tauri::Emitter;
    let cfg = state.config.lock().await.clone();
    let binary = cfg.ollama_binary_or_default();
    let mut cmd = if cfg!(target_os = "windows") {
        let mut c = tokio::process::Command::new("cmd");
        c.arg("/C").arg(binary);
        c
    } else {
        tokio::process::Command::new(binary)
    };
    cmd.arg("pull").arg(&name);
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null());
    let mut child = cmd.spawn().map_err(|e| format!("spawn ollama pull: {e}"))?;

    // Ollama streams progress on stderr as NDJSON like
    // {"status":"pulling ...","completed":N,"total":M}. Forward each line,
    // tagged with the pull's id (the model name) so concurrent pulls each
    // drive their own bar in the Models panel.
    let stderr = child.stderr.take();
    if let Some(stderr) = stderr {
        let app2 = app.clone();
        let id2 = name.clone();
        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let payload = serde_json::json!({ "id": id2, "line": line });
                let _ = app2.emit("ollama-pull-progress", payload);
            }
        });
    }
    // Also capture stdout for completeness.
    let stdout = child.stdout.take();
    if let Some(stdout) = stdout {
        let app2 = app.clone();
        let id2 = name.clone();
        tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let payload = serde_json::json!({ "id": id2, "line": line });
                let _ = app2.emit("ollama-pull-progress", payload);
            }
        });
    }

    let outcome: Result<String, String> = async {
        let status = child
            .wait()
            .await
            .map_err(|e| format!("ollama pull wait: {e}"))?;
        if !status.success() {
            return Err(format!("ollama pull exited with {status}"));
        }
        Ok(name.clone())
    }
    .await;
    {
        use tauri::Emitter;
        let payload = match &outcome {
            Ok(_) => serde_json::json!({ "id": name, "phase": "done" }),
            Err(e) => serde_json::json!({ "id": name, "phase": "error", "error": e }),
        };
        let _ = app.emit("ollama-pull-progress", payload);
    }
    outcome
}

/// Delete an Ollama model (the models panel's bin button) via the server's
/// `DELETE /api/delete`. Frees the model's blobs from Ollama's store.
#[tauri::command]
pub async fn delete_ollama_model(
    state: State<'_, WebState>,
    name: String,
) -> Result<(), String> {
    let cfg = state.config.lock().await;
    let url = format!("{}/api/delete", cfg.ollama_url.trim_end_matches('/'));
    drop(cfg);
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .delete(&url)
        .json(&serde_json::json!({ "model": name }))
        .send()
        .await
        .map_err(|e| format!("Ollama not running? DELETE {url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("DELETE {url} returned {}", resp.status()));
    }
    Ok(())
}

/// Auto-install Ollama on Windows by running the bundled installer. Returns the
/// path to the installer that was (or would be) run.
#[tauri::command]
pub async fn install_ollama(state: State<'_, WebState>) -> Result<String, String> {
    // The bundled installer ships under <exe_dir>/resources or the deps folder.
    // Fall back to downloading from ollama.com if not found.
    let candidate_dirs = [
        state.paths.data_dir.join("ollama"),
        std::env::current_exe()
            .map_err(|e| e.to_string())?
            .parent()
            .map(|p| p.join("resources"))
            .unwrap_or_else(|| std::path::PathBuf::from(".")),
    ];
    let mut installer: Option<std::path::PathBuf> = None;
    for d in &candidate_dirs {
        let f = d.join("OllamaSetup.exe");
        if f.is_file() {
            installer = Some(f);
            break;
        }
    }
    let installer = match installer {
        Some(p) => p,
        None => {
            // Download the installer to the data dir.
            let dest = state.paths.data_dir.join("OllamaSetup.exe");
            let url = "https://ollama.com/download/OllamaSetup.exe";
            let bytes = reqwest::get(url)
                .await
                .map_err(|e| format!("download installer: {e}"))?
                .bytes()
                .await
                .map_err(|e| format!("read installer: {e}"))?;
            std::fs::write(&dest, &bytes).map_err(|e| format!("write installer: {e}"))?;
            dest
        }
    };

    // Run the installer silently. NSIS installers accept /S for silent mode.
    let mut cmd = tokio::process::Command::new(&installer);
    cmd.arg("/S");
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let out = tokio::time::timeout(std::time::Duration::from_secs(180), cmd.output())
        .await
        .map_err(|_| "installer timed out".to_string())?
        .map_err(|e| format!("run installer: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("installer failed: {stderr}"));
    }
    Ok(installer.display().to_string())
}

/// Start the Ollama server (stopping any other backend), switch to it, and set
/// the active model. The "Run" semantics for the Ollama box.
#[tauri::command]
pub async fn run_ollama(
    app: AppHandle,
    state: State<'_, WebState>,
    model: String,
) -> Result<(), String> {
    let cfg = state.config.lock().await.clone();
    let binary = cfg.ollama_binary_or_default();
    state
        .process_mgr
        .start_ollama(Some(binary))
        .await
        .map_err(|e| e.to_string())?;
    let local_url = {
        let mut c = state.config.lock().await;
        c.active_backend = "ollama".into();
        c.active_provider_id = None;
        let clone = c.clone();
        drop(c);
        save_config(&state.paths, &clone).map_err(|e| format!("Save config: {e}"))?;
        clone.ollama_url.clone()
    };
    state.provider.set_local(LocalBackend::Ollama, local_url).await;
    apply_model(&app, &state, model).await?;
    Ok(())
}

// ---- Provider API (red box) -----------------------------------------------

/// List registered cloud providers (API keys masked).
#[tauri::command]
pub async fn list_providers(state: State<'_, WebState>) -> Result<Vec<Provider>, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.list_providers().map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// Register a new cloud provider. Returns its id.
#[tauri::command]
pub async fn create_provider(
    state: State<'_, WebState>,
    name: String,
    base_url: String,
    api_key: String,
) -> Result<i64, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.create_provider(&name, &base_url, &api_key, "openai")
                .map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// Update a registered provider's fields.
#[tauri::command]
pub async fn update_provider(
    state: State<'_, WebState>,
    id: i64,
    name: String,
    base_url: String,
    api_key: String,
) -> Result<(), String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.update_provider(id, &name, &base_url, &api_key, "openai")
                .map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// Delete a registered provider. Clears the active selection if it was active.
#[tauri::command]
pub async fn delete_provider(state: State<'_, WebState>, id: i64) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        match store.as_ref() {
            Some(s) => {
                let s = s.lock().await;
                s.delete_provider(id).map_err(|e| e.to_string())?;
            }
            None => return Err("Not unlocked yet".into()),
        }
    }
    // Clear the active selection if this provider was active.
    let mut cfg = state.config.lock().await;
    if cfg.active_provider_id == Some(id) {
        cfg.active_provider_id = None;
        let clone = cfg.clone();
        drop(cfg);
        save_config(&state.paths, &clone).map_err(|e| format!("Save config: {e}"))?;
        // Revert the route to the active local backend.
        let local_backend = LocalBackend::parse(&clone.active_backend)
            .unwrap_or(LocalBackend::Ollama);
        state
            .provider
            .set_local(local_backend, clone.resolved_provider_url())
            .await;
    }
    Ok(())
}

/// Read a provider's cleartext API key (for the panel's hover-reveal).
#[tauri::command]
pub async fn get_provider_key(
    state: State<'_, WebState>,
    id: i64,
) -> Result<String, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            match s.get_provider_endpoint(id).map_err(|e| e.to_string())? {
                Some((_, key)) => Ok(key),
                None => Err("Provider not found".into()),
            }
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// Switch the active route to a cloud provider and set the model. The "Run"
/// semantics for the red box.
#[tauri::command]
pub async fn run_provider(
    app: AppHandle,
    state: State<'_, WebState>,
    provider_id: i64,
    model: Option<String>,
) -> Result<(), String> {
    // 1. Stop any local server (cloud needs none).
    state.process_mgr.stop_all().await;
    // 2. Load the provider endpoint.
    let (base_url, key) = {
        let store = state.store.lock().await;
        let s = store.as_ref().ok_or("Not unlocked yet")?.clone();
        let s = s.lock().await;
        s.get_provider_endpoint(provider_id)
            .map_err(|e| e.to_string())?
            .ok_or("Provider not found")?
    };
    // 3. Switch the route.
    state
        .provider
        .set_cloud(CloudRoute {
            provider_id,
            base_url: base_url.clone(),
            api_key: key,
        })
        .await;
    // 4. Persist the active provider.
    {
        let mut cfg = state.config.lock().await;
        cfg.active_provider_id = Some(provider_id);
        let clone = cfg.clone();
        drop(cfg);
        save_config(&state.paths, &clone).map_err(|e| format!("Save config: {e}"))?;
    }
    // 5. Resolve + set the model: explicit arg, else the persisted model, else
    //    the first model from the provider's /v1/models.
    let model = match model {
        Some(m) if !m.trim().is_empty() => m,
        _ => {
            let persisted = state.config.lock().await.model.clone();
            if !persisted.is_empty() {
                persisted
            } else {
                // Pick the first available model from the provider.
                match state.provider.list_models().await {
                    Ok(list) if !list.is_empty() => list.into_iter().next().unwrap(),
                    _ => "gpt-4o-mini".to_string(),
                }
            }
        }
    };
    apply_model(&app, &state, model).await?;
    Ok(())
}

/// Token consumption (in + out) for a provider in the last hour.
#[tauri::command]
pub async fn provider_usage_last_hour(
    state: State<'_, WebState>,
    provider_id: i64,
) -> Result<i64, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.provider_usage_last_hour(provider_id)
                .map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// Get the latest health snapshot.
#[tauri::command]
pub async fn get_health(state: State<'_, WebState>) -> Result<health::HealthState, String> {
    Ok(state.health.lock().await.clone())
}

/// Live runtime metrics (T/s, TTFT, TBT, busy) measured at the dispatch layer
/// for the health bar. Works for every backend, not just AmberCore.
#[tauri::command]
pub async fn get_runtime_metrics(
    state: State<'_, WebState>,
) -> Result<crate::model::ProviderStats, String> {
    Ok(state.provider.merged_stats().await)
}

// ---- Science Workbench: models, profiles, workdir ---------------------

/// Switch the active model live. Persists the choice to `config.toml` so it
/// survives restarts, notifies the health monitor (via the watch channel), and
/// tells the agent runtime to use the new model from the next turn.
#[tauri::command]
pub async fn set_model(
    app: AppHandle,
    state: State<'_, WebState>,
    model: String,
) -> Result<(), String> {
    apply_model(&app, &state, model).await
}

/// Shared model-switch implementation (used by `set_model` and the `run_*`
/// commands). Persists the model, notifies the health monitor + runtime, and
/// emits `model-changed` so every model selector in the UI stays in sync.
async fn apply_model(app: &AppHandle, state: &WebState, model: String) -> Result<(), String> {
    // 1. Persist to config.toml.
    {
        let mut cfg = state.config.lock().await;
        cfg.model = model.clone();
        let cfg_clone = cfg.clone();
        drop(cfg);
        save_config(&state.paths, &cfg_clone).map_err(|e| format!("Save config: {e}"))?;
    }

    // 2. Update the health monitor's view of the active model.
    if let Some(tx) = state.model_tx.lock().await.as_ref() {
        let _ = tx.send(model.clone());
    }

    // 3. Tell the runtime to use the new model.
    let tx = state.cmd_tx.lock().await;
    match tx.as_ref() {
        Some(sender) => sender
            .send(Command::SetModel { model })
            .await
            .map_err(|e| format!("Agent channel closed: {e}"))?,
        None => return Err("Not unlocked yet".into()),
    }
    drop(tx);

    // 4. Notify the UI: the chat model selector and the Models panel both
    //    re-render from this one event (two-way sync).
    let info = active_route_info(state).await;
    let _ = app.emit(MODEL_CHANGED_EVENT, &info);
    Ok(())
}

/// List all profiles (for the sidebar profile selector).
#[tauri::command]
pub async fn list_profiles(state: State<'_, WebState>) -> Result<Vec<Profile>, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.list_profiles().map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// Create a new profile with default behavior settings. Returns its id.
#[tauri::command]
pub async fn create_profile(
    state: State<'_, WebState>,
    name: String,
) -> Result<i64, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.create_profile(&name, "writes_only", 25, 50)
                .map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// Switch the active profile. Persists the choice and pushes the profile's
/// behavior settings to the agent runtime live.
#[tauri::command]
pub async fn switch_profile(
    state: State<'_, WebState>,
    id: i64,
) -> Result<Profile, String> {
    let profile = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        let p = s.get_profile(id).map_err(|e| e.to_string())?;
        let p = p.ok_or("Profile not found")?;
        s.set_active_profile_id(id).map_err(|e| e.to_string())?;
        p
    };

    // Push the profile's behavior to the runtime.
    let tx = state.cmd_tx.lock().await;
    if let Some(sender) = tx.as_ref() {
        sender
            .send(Command::ApplyProfileSettings {
                approval_policy: parse_approval_policy(&profile.approval_policy),
                max_iterations: profile.max_iterations as u32,
                context_window: profile.context_window as u32,
            })
            .await
            .map_err(|e| format!("Agent channel closed: {e}"))?;
    }
    drop(tx);

    // Apply the profile's OWN working directory (profile-scoped since the
    // sidebar rework). A profile with no directory chosen leaves the runtime's
    // current cwd in place — the UI's placeholder drives the user to pick one.
    let profile_wd = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.get_profile_workdir(id)
            .map_err(|e| e.to_string())?
            .filter(|p| !p.is_empty())
    };
    if let Some(p) = profile_wd {
        let workdir = std::path::PathBuf::from(&p);
        *state.workdir.lock().await = workdir.clone();
        let tx = state.cmd_tx.lock().await;
        if let Some(sender) = tx.as_ref() {
            let _ = sender.send(Command::SetWorkdir { workdir }).await;
        }
    }
    Ok(profile)
}

/// Get the ACTIVE PROFILE's working directory for display — empty string when
/// the profile has none chosen yet (the UI shows its "select a directory"
/// placeholder). The runtime keeps a cwd fallback for execution.
#[tauri::command]
pub async fn get_workdir(state: State<'_, WebState>) -> Result<String, String> {
    let store = state.store.lock().await;
    let Some(s) = store.as_ref() else {
        return Ok(String::new());
    };
    let s = s.lock().await;
    let id = s
        .get_active_profile_id()
        .map_err(|e| e.to_string())?
        .ok_or("No profiles")?;
    if let Some(p) = s.get_profile_workdir(id).map_err(|e| e.to_string())? {
        return Ok(p);
    }
    // Lazy one-time migration: attribute the legacy global workdir to the
    // Default profile so existing installs keep their directory.
    let is_default = s
        .get_profile(id)
        .map_err(|e| e.to_string())?
        .map(|p| p.is_default)
        .unwrap_or(false);
    if is_default {
        if let Some(legacy) = s.get_workdir().map_err(|e| e.to_string())? {
            if !legacy.is_empty() {
                let _ = s.set_profile_workdir(id, &legacy);
                return Ok(legacy);
            }
        }
    }
    Ok(String::new())
}

/// Change the ACTIVE PROFILE's working directory live. Persists it under the
/// profile and notifies the runtime so it applies from the next user turn.
#[tauri::command]
pub async fn set_workdir(
    state: State<'_, WebState>,
    path: String,
) -> Result<(), String> {
    let workdir = std::path::PathBuf::from(&path);
    *state.workdir.lock().await = workdir.clone();

    // Persist under the active profile (+ keep the legacy global in sync so
    // pre-unlock boot resolution stays deterministic).
    {
        let store = state.store.lock().await;
        if let Some(s) = store.as_ref() {
            let s = s.lock().await;
            if let Some(id) = s.get_active_profile_id().map_err(|e| e.to_string())? {
                let _ = s.set_profile_workdir(id, &path);
                let _ = s.set_workdir(&path);
            }
        }
    }

    // Notify the runtime.
    let tx = state.cmd_tx.lock().await;
    if let Some(sender) = tx.as_ref() {
        sender
            .send(Command::SetWorkdir { workdir })
            .await
            .map_err(|e| format!("Agent channel closed: {e}"))?;
    }
    Ok(())
}

// ---- Sidebar console -------------------------------------------------------

/// Output of one sidebar-console command.
#[derive(serde::Serialize)]
pub struct ConsoleOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

/// Run a one-shot shell command in the active workdir for the SIDEBAR CONSOLE.
/// This is the USER typing — not the agent acting — so there is no approval
/// gate. Mirrors the agent's shell tool's execution shape: `cmd /C` on Windows
/// / `sh -c` elsewhere, combined-capture stdout+stderr, 60 s timeout, 20k-char
/// cap per stream.
#[tauri::command]
pub async fn console_run(
    state: State<'_, WebState>,
    command: String,
) -> Result<ConsoleOutput, String> {
    const TIMEOUT_SECS: u64 = 60;
    const MAX_CHARS: usize = 20_000;
    let cwd = state.workdir.lock().await.clone();
    let (program, flag) = if cfg!(target_os = "windows") {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    };
    let mut cmd = tokio::process::Command::new(program);
    cmd.arg(flag).arg(&command).current_dir(&cwd);
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null());
    let child = cmd.spawn().map_err(|e| format!("spawn {program}: {e}"))?;
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(TIMEOUT_SECS),
        child.wait_with_output(),
    )
    .await
    .map_err(|_| format!("command timed out after {TIMEOUT_SECS}s and was killed"))?
    .map_err(|e| format!("wait: {e}"))?;
    let clip = |raw: &[u8]| -> String {
        let mut text = String::from_utf8_lossy(raw).into_owned();
        text.truncate(MAX_CHARS);
        text
    };
    Ok(ConsoleOutput {
        stdout: clip(&out.stdout),
        stderr: clip(&out.stderr),
        exit_code: out.status.code().unwrap_or(-1),
    })
}

// ---- Science Workbench: skills (Panel 2) ---------------------------------

/// Resolve the active profile id from the store, or None if not unlocked.
async fn active_profile_id(state: &State<'_, WebState>) -> Result<i64, String> {
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.get_active_profile_id()
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "No active profile".to_string())
}

/// Load the enabled skills for the active profile and push them to the runtime
/// so the system prompt rebuilds. Shared by enable/disable/install/create/edit.
async fn reload_skills(state: &State<'_, WebState>) -> Result<(), String> {
    let pid = active_profile_id(state).await?;
    let skills: Vec<(String, String, String)> = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.list_enabled_skills_for_profile(pid)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|sk| (sk.name, sk.description, sk.body))
            .collect()
    };
    let tx = state.cmd_tx.lock().await;
    if let Some(sender) = tx.as_ref() {
        sender
            .send(Command::ReloadSkills { skills })
            .await
            .map_err(|e| format!("Agent channel closed: {e}"))?;
    }
    Ok(())
}

/// List all skills.
#[tauri::command]
pub async fn list_skills(state: State<'_, WebState>) -> Result<Vec<Skill>, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.list_skills().map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// List every skill with its enabled state for the active profile.
#[tauri::command]
pub async fn list_skills_for_active_profile(
    state: State<'_, WebState>,
) -> Result<Vec<ProfileSkill>, String> {
    let pid = active_profile_id(&state).await?;
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.list_skills_for_profile(pid).map_err(|e| e.to_string())
}

/// Create a new locally-authored skill, then reload the prompt.
#[tauri::command]
pub async fn create_skill(
    state: State<'_, WebState>,
    name: String,
    description: String,
    body: String,
) -> Result<i64, String> {
    let id = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.create_skill(&name, &description, &body, "local", None)
            .map_err(|e| e.to_string())?
    };
    reload_skills(&state).await?;
    Ok(id)
}

/// Update a skill's editable fields, then reload the prompt.
#[tauri::command]
pub async fn update_skill(
    state: State<'_, WebState>,
    id: i64,
    name: String,
    description: String,
    body: String,
) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.update_skill(id, &name, &description, &body)
            .map_err(|e| e.to_string())?;
    }
    reload_skills(&state).await?;
    Ok(())
}

/// Delete a skill, then reload the prompt.
#[tauri::command]
pub async fn delete_skill(state: State<'_, WebState>, id: i64) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.delete_skill(id).map_err(|e| e.to_string())?;
    }
    reload_skills(&state).await?;
    Ok(())
}

/// Enable/disable a skill for the active profile, then reload the prompt.
#[tauri::command]
pub async fn set_skill_enabled(
    state: State<'_, WebState>,
    skill_id: i64,
    enabled: bool,
) -> Result<(), String> {
    let pid = active_profile_id(&state).await?;
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.set_skill_enabled_for_profile(pid, skill_id, enabled)
            .map_err(|e| e.to_string())?;
    }
    reload_skills(&state).await?;
    Ok(())
}

/// Search GitHub for markdown files matching a query (candidate skills).
#[tauri::command]
pub async fn search_github_skills(query: String) -> Result<Vec<GithubSkillHit>, String> {
    skill_helpers::search_github(&query)
        .await
        .map_err(|e| e.to_string())
}

/// Install a skill from a GitHub raw URL: fetch its markdown, store it, enable
/// it for the active profile, and reload the prompt.
#[tauri::command]
pub async fn install_github_skill(
    state: State<'_, WebState>,
    name: String,
    description: String,
    raw_url: String,
) -> Result<i64, String> {
    let body = skill_helpers::fetch_raw(&raw_url)
        .await
        .map_err(|e| e.to_string())?;
    let pid = active_profile_id(&state).await?;
    let id = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        let id = s
            .create_skill(&name, &description, &body, "github", Some(&raw_url))
            .map_err(|e| e.to_string())?;
        // Enable for the active profile (this also marks it customized).
        s.set_skill_enabled_for_profile(pid, id, true)
            .map_err(|e| e.to_string())?;
        id
    };
    reload_skills(&state).await?;
    Ok(id)
}

// ---- Science Workbench: tools (Panel 3) ---------------------------------

/// Reload the enabled user tools for the active profile and push them to the
/// runtime, which rebuilds the `ToolRegistry` + the system prompt. Shared by
/// create/update/delete/set_tool_enabled.
async fn reload_tools(state: &State<'_, WebState>) -> Result<(), String> {
    let pid = active_profile_id(state).await?;
    let specs: Vec<ToolSpec> = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        ToolSpec::from_rows(
            &s.list_enabled_tools_for_profile(pid)
                .map_err(|e| e.to_string())?,
        )
    };
    let tx = state.cmd_tx.lock().await;
    if let Some(sender) = tx.as_ref() {
        sender
            .send(Command::ReloadTools { tools: specs })
            .await
            .map_err(|e| format!("Agent channel closed: {e}"))?;
    }
    Ok(())
}

/// List all user tools.
#[tauri::command]
pub async fn list_tools(state: State<'_, WebState>) -> Result<Vec<ToolRow>, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.list_tools().map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// List every tool with its enabled state for the active profile.
#[tauri::command]
pub async fn list_tools_for_active_profile(
    state: State<'_, WebState>,
) -> Result<Vec<ProfileTool>, String> {
    let pid = active_profile_id(&state).await?;
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.list_tools_for_profile(pid).map_err(|e| e.to_string())
}

/// Create a new locally-authored tool, then reload the registry.
#[tauri::command]
pub async fn create_tool(
    state: State<'_, WebState>,
    name: String,
    description: String,
    interpreter: String,
    script_body: String,
    params_schema: String,
    tool_kind: String,
) -> Result<i64, String> {
    let id = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.create_tool(
            &name,
            &description,
            &interpreter,
            &script_body,
            &params_schema,
            &tool_kind,
            "local",
            None,
        )
        .map_err(|e| e.to_string())?
    };
    reload_tools(&state).await?;
    Ok(id)
}

/// Update a tool's editable fields, then reload the registry.
#[tauri::command]
pub async fn update_tool(
    state: State<'_, WebState>,
    id: i64,
    name: String,
    description: String,
    interpreter: String,
    script_body: String,
    params_schema: String,
    tool_kind: String,
) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.update_tool(
            id,
            &name,
            &description,
            &interpreter,
            &script_body,
            &params_schema,
            &tool_kind,
        )
        .map_err(|e| e.to_string())?;
    }
    reload_tools(&state).await?;
    Ok(())
}

/// Delete a tool, then reload the registry.
#[tauri::command]
pub async fn delete_tool(state: State<'_, WebState>, id: i64) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.delete_tool(id).map_err(|e| e.to_string())?;
    }
    reload_tools(&state).await?;
    Ok(())
}

/// Enable/disable a tool for the active profile, then reload the registry.
#[tauri::command]
pub async fn set_tool_enabled(
    state: State<'_, WebState>,
    tool_id: i64,
    enabled: bool,
) -> Result<(), String> {
    let pid = active_profile_id(&state).await?;
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.set_tool_enabled_for_profile(pid, tool_id, enabled)
            .map_err(|e| e.to_string())?;
    }
    reload_tools(&state).await?;
    Ok(())
}

/// Fetch a tool script body from a GitHub raw URL (so the frontend can populate
/// the edit form). The user fills in name/description/interpreter/schema/kind,
/// then calls `create_tool`. We restrict to common script extensions.
#[tauri::command]
pub async fn prefetch_github_tool(
    raw_url: String,
) -> Result<String, String> {
    skill_helpers::fetch_raw(&raw_url)
        .await
        .map_err(|e| e.to_string())
}

/// Search GitHub for script files (candidate tools). Reuses the skills search
/// helper with a script-extensions query.
#[tauri::command]
pub async fn search_github_tools(query: String) -> Result<Vec<GithubSkillHit>, String> {
    // Restrict to common script file extensions so results are plausibly tools.
    let q = format!("{query} extension:py OR extension:js OR extension:sh OR extension:ps1");
    skill_helpers::search_github(&q)
        .await
        .map_err(|e| e.to_string())
}

// ---- Science Workbench: context (Panel 4) -------------------------------

/// Reload the enabled context files for the active profile and push them to the
/// runtime, which rebuilds the system prompt's `## Project Context` section.
async fn reload_context(state: &State<'_, WebState>) -> Result<(), String> {
    let pid = active_profile_id(state).await?;
    let context: Vec<(String, String, String)> = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.list_enabled_context_for_profile(pid)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|c| (c.name, c.description, c.body))
            .collect()
    };
    let tx = state.cmd_tx.lock().await;
    if let Some(sender) = tx.as_ref() {
        sender
            .send(Command::ReloadContext { context })
            .await
            .map_err(|e| format!("Agent channel closed: {e}"))?;
    }
    Ok(())
}

/// List all context files.
#[tauri::command]
pub async fn list_context(state: State<'_, WebState>) -> Result<Vec<ContextFile>, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.list_context().map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// List every context file with its enabled state for the active profile.
#[tauri::command]
pub async fn list_context_for_active_profile(
    state: State<'_, WebState>,
) -> Result<Vec<ProfileContext>, String> {
    let pid = active_profile_id(&state).await?;
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.list_context_for_profile(pid).map_err(|e| e.to_string())
}

/// Create a new context file, then reload the prompt.
#[tauri::command]
pub async fn create_context(
    state: State<'_, WebState>,
    name: String,
    description: String,
    body: String,
) -> Result<i64, String> {
    let id = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.create_context(&name, &description, &body)
            .map_err(|e| e.to_string())?
    };
    reload_context(&state).await?;
    Ok(id)
}

/// Update a context file, then reload the prompt.
#[tauri::command]
pub async fn update_context(
    state: State<'_, WebState>,
    id: i64,
    name: String,
    description: String,
    body: String,
) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.update_context(id, &name, &description, &body)
            .map_err(|e| e.to_string())?;
    }
    reload_context(&state).await?;
    Ok(())
}

/// Delete a context file, then reload the prompt.
#[tauri::command]
pub async fn delete_context(state: State<'_, WebState>, id: i64) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.delete_context(id).map_err(|e| e.to_string())?;
    }
    reload_context(&state).await?;
    Ok(())
}

/// Enable/disable a context file for the active profile, then reload the prompt.
#[tauri::command]
pub async fn set_context_enabled(
    state: State<'_, WebState>,
    context_id: i64,
    enabled: bool,
) -> Result<(), String> {
    let pid = active_profile_id(&state).await?;
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.set_context_enabled_for_profile(pid, context_id, enabled)
            .map_err(|e| e.to_string())?;
    }
    reload_context(&state).await?;
    Ok(())
}

// ---- Science Workbench: memory / MCP (Panel 5) --------------------------

/// The result of a `test_memory_connection` probe — shown in the panel UI.
#[derive(Debug, Serialize)]
pub struct MemoryTestResult {
    pub ok: bool,
    pub tool_count: usize,
    pub error: Option<String>,
}

/// Reload the enabled MCP connections for the active profile and push them to
/// the runtime, which reconnects and rebuilds the tool registry.
async fn reload_mcp(state: &State<'_, WebState>) -> Result<(), String> {
    let pid = active_profile_id(state).await?;
    let connections: Vec<McpConnectionSpec> = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.list_enabled_memory_for_profile(pid)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|m| McpConnectionSpec {
                id: m.id,
                name: m.name,
                transport: m.transport,
                command: m.command,
                args_json: m.args_json,
            })
            .collect()
    };
    let tx = state.cmd_tx.lock().await;
    if let Some(sender) = tx.as_ref() {
        sender
            .send(Command::ReloadMcp { connections })
            .await
            .map_err(|e| format!("Agent channel closed: {e}"))?;
    }
    Ok(())
}

/// List all memory sources (MCP connections).
#[tauri::command]
pub async fn list_memory(state: State<'_, WebState>) -> Result<Vec<crate::db::MemorySource>, String> {
    let store = state.store.lock().await;
    match store.as_ref() {
        Some(s) => {
            let s = s.lock().await;
            s.list_memory().map_err(|e| e.to_string())
        }
        None => Err("Not unlocked yet".into()),
    }
}

/// List every memory source with its enabled state for the active profile.
#[tauri::command]
pub async fn list_memory_for_active_profile(
    state: State<'_, WebState>,
) -> Result<Vec<crate::db::ProfileMemory>, String> {
    let pid = active_profile_id(&state).await?;
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.list_memory_for_profile(pid).map_err(|e| e.to_string())
}

/// Create a new MCP connection, then reload the runtime.
#[tauri::command]
pub async fn create_memory(
    state: State<'_, WebState>,
    name: String,
    description: String,
    transport: String,
    command: String,
    args_json: String,
) -> Result<i64, String> {
    let id = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.create_memory(&name, &description, &transport, &command, &args_json)
            .map_err(|e| e.to_string())?
    };
    reload_mcp(&state).await?;
    Ok(id)
}

/// Update an MCP connection, then reload the runtime.
#[tauri::command]
pub async fn update_memory(
    state: State<'_, WebState>,
    id: i64,
    name: String,
    description: String,
    transport: String,
    command: String,
    args_json: String,
) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.update_memory(id, &name, &description, &transport, &command, &args_json)
            .map_err(|e| e.to_string())?;
    }
    reload_mcp(&state).await?;
    Ok(())
}

/// Delete an MCP connection, then reload the runtime.
#[tauri::command]
pub async fn delete_memory(state: State<'_, WebState>, id: i64) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.delete_memory(id).map_err(|e| e.to_string())?;
    }
    reload_mcp(&state).await?;
    Ok(())
}

/// Enable/disable an MCP connection for the active profile, then reload.
#[tauri::command]
pub async fn set_memory_enabled(
    state: State<'_, WebState>,
    memory_id: i64,
    enabled: bool,
) -> Result<(), String> {
    let pid = active_profile_id(&state).await?;
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.set_memory_enabled_for_profile(pid, memory_id, enabled)
            .map_err(|e| e.to_string())?;
    }
    reload_mcp(&state).await?;
    Ok(())
}

/// Probe a (possibly unsaved) MCP connection: connect, list tools, disconnect.
/// Lets the UI validate before saving. Never fails — returns an error string.
#[tauri::command]
pub async fn test_memory_connection(
    _state: State<'_, WebState>,
    transport: String,
    command: String,
    args_json: String,
) -> Result<MemoryTestResult, String> {
    let spec = McpConnectionSpec {
        id: 0,
        name: "<test>".into(),
        transport,
        command,
        args_json,
    };
    let t = spec.to_transport();
    match mcp_helpers::McpClient::connect(t).await {
        Ok(client) => match client.list_tools().await {
            Ok(tools) => Ok(MemoryTestResult {
                ok: true,
                tool_count: tools.len(),
                error: None,
            }),
            Err(e) => Ok(MemoryTestResult {
                ok: false,
                tool_count: 0,
                error: Some(format!("Connected but tools/list failed: {e}")),
            }),
        },
        Err(e) => Ok(MemoryTestResult {
            ok: false,
            tool_count: 0,
            error: Some(e.to_string()),
        }),
    }
}

// ---- Security: passphrase change + TOTP 2FA (Panel: security menu) ------

/// Whether 2FA is enabled (read from the unencrypted marker file, so the unlock
/// screen can decide whether to show the code field without opening the DB).
#[tauri::command]
pub async fn has_totp(state: State<'_, WebState>) -> Result<bool, String> {
    if std::fs::read_to_string(&state.paths.totp_flag_path)
        .map(|s| s.trim() == "1")
        .unwrap_or(false)
    {
        return Ok(true);
    }
    // After a wiped v2 exit the staging flag is gone — the tail header's
    // bundle still says whether a recovery wrap exists (2FA on), so the
    // unlock screen can keep offering recovery.
    Ok(load_bundle_anywhere(&state)?.is_some_and(|b| b.has_recovery()))
}

/// Begin 2FA setup: generate a TOTP secret + otpauth URL for the user to add to
/// their authenticator (Proton Pass, 2FAS, etc.). The secret is held in the
/// encrypted DB as a "pending" setting until [`confirm_totp`] verifies a live
/// code, so an abandoned setup never enables 2FA.
#[tauri::command]
pub async fn setup_totp(
    state: State<'_, WebState>,
    account: String,
) -> Result<totp_lib::TotpSetup, String> {
    let instance = totp_lib::generate(&account).map_err(|e| e.to_string())?;
    let setup = instance.to_setup();

    // Stash the pending secret in the encrypted DB (NOT enabled yet).
    let store = state.store.lock().await;
    let store = store.as_ref().ok_or("Not unlocked yet")?;
    let s = store.lock().await;
    s.set_setting("totp_pending_secret_b32", &setup.secret_b32)
        .map_err(|e| e.to_string())?;
    s.set_setting("totp_pending_account", &account)
        .map_err(|e| e.to_string())?;

    Ok(setup)
}

/// Confirm a 2FA setup by checking the user's first code, then persist the
/// config as active. After this, unlock requires a code.
#[tauri::command]
pub async fn confirm_totp(
    state: State<'_, WebState>,
    code: String,
) -> Result<(), String> {
    let (secret, account) = {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        let secret = s
            .get_setting("totp_pending_secret_b32")
            .map_err(|e| e.to_string())?
            .ok_or("No pending 2FA setup. Start setup first.")?;
        let account = s
            .get_setting("totp_pending_account")
            .map_err(|e| e.to_string())?
            .unwrap_or_default();
        (secret, account)
    };

    let totp = totp_lib::from_secret(&secret, &account).map_err(|e| e.to_string())?;
    if !totp_lib::verify(&totp, &code) {
        return Err("Code doesn't match. Make sure your authenticator's time is correct and try again.".into());
    }

    // Promote pending → active, and flip the unencrypted flag.
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.set_totp_config(&secret, &account).map_err(|e| e.to_string())?;
        // Clear the pending rows.
        let _ = s.set_setting("totp_pending_secret_b32", "");
        let _ = s.set_setting("totp_pending_account", "");
    }
    std::fs::write(&state.paths.totp_flag_path, "1")
        .map_err(|e| format!("write 2fa flag: {e}"))?;

    // Establish the recovery wraps: encrypt the DB key AND the capsule key
    // under a key derived from the TOTP seed, so a forgotten launch password
    // can be recovered via a current TOTP code. The seed is stored in the
    // bundle for verification.
    {
        let db_key = state
            .db_key
            .lock()
            .await
            .clone()
            .ok_or("Not unlocked yet")?;
        let mut bundle = load_bundle_anywhere(&state)?
            .ok_or("No key bundle found — the app is not initialized.")?;
        bundle
            .set_recovery(&db_key, &secret)
            .map_err(|e| format!("set recovery wrap: {e}"))?;
        if let Some(k) = state.capsule_key.lock().await.clone() {
            bundle
                .set_recovery_capsule(&k)
                .map_err(|e| format!("set capsule recovery wrap: {e}"))?;
        }
        bundle
            .save(&state.paths.key_bundle_path)
            .map_err(|e| format!("save key bundle: {e}"))?;
    }
    Ok(())
}

/// Disable 2FA: remove the TOTP config, clear the marker file, and remove the
/// recovery wrap from the key bundle (so the lost-launch-password recovery path
/// no longer works). The caller must already be unlocked (so the DB is open).
#[tauri::command]
pub async fn disable_totp(state: State<'_, WebState>) -> Result<(), String> {
    {
        let store = state.store.lock().await;
        let store = store.as_ref().ok_or("Not unlocked yet")?;
        let s = store.lock().await;
        s.clear_totp_config().map_err(|e| e.to_string())?;
    }
    let _ = std::fs::remove_file(&state.paths.totp_flag_path);
    // Drop the recovery wrap (it was keyed by the 2FA seed we just removed).
    if state.paths.key_bundle_path.exists() {
        let mut bundle = KeyBundle::load(&state.paths.key_bundle_path)
            .map_err(|e| format!("Read key bundle to clear recovery: {e}"))?;
        bundle.clear_recovery();
        bundle
            .save(&state.paths.key_bundle_path)
            .map_err(|e| format!("save key bundle: {e}"))?;
    }
    Ok(())
}

/// Change the **database access password** (the secret that derives the
/// SQLCipher key) in place via `PRAGMA rekey`.
///
/// This is the secure anchor password — set once at first run (default
/// "PhoenixAgent") and used autonomously by the app thereafter. The user changes
/// it here only if they want to rotate the encryption key. The flow: verify the
/// current DB password, rotate the salt, derive the new key, re-encrypt every DB
/// page, then **re-wrap the new key under the launch password** so the on-disk
/// key bundle (`keys.phx`) stays consistent and unlock keeps working. The launch
/// password is required as a parameter for that re-wrap.
///
/// TOTP no longer folds into the DB key (it gates unlock recovery, not the key),
/// so 2FA state is unaffected by a DB password change. If a recovery wrap
/// existed (2FA on), it is refreshed under the new key + the stored seed.
#[tauri::command]
pub async fn change_passphrase(
    app: AppHandle,
    state: State<'_, WebState>,
    current_db_password: String,
    new_db_password: String,
    confirm: String,
    launch_password: String,
) -> Result<(), String> {
    if new_db_password.len() < 8 {
        return Err("New database password must be at least 8 characters.".into());
    }
    if new_db_password != confirm {
        return Err("New database passwords do not match.".into());
    }

    // 1. Verify the current DB password by opening a fresh connection.
    let salt = load_or_create_salt(&state.paths.salt_path)
        .map_err(|e| format!("Salt error: {e}"))?;
    let old_key = derive_key(&current_db_password, &salt, None)
        .map_err(|e| format!("Key derivation error: {e}"))?;
    let conn = open_encrypted(&state.paths.db_path, &old_key)
        .map_err(|_| "Current database password is incorrect.".to_string())?;

    // 2. Rotate the salt and derive the new key.
    let new_salt = rotate_salt(&state.paths.salt_path)
        .map_err(|e| format!("rotate salt: {e}"))?;
    let new_key = derive_key(&new_db_password, &new_salt, None)
        .map_err(|e| format!("new key derivation: {e}"))?;

    // 3. Re-encrypt the DB in place with the new raw key (wrapped x'<hex>' form).
    let _ = conn.pragma_update(None, "wal_checkpoint", "TRUNCATE");
    let new_pragma = format!("x'{}'", new_key.to_hex());
    conn.pragma_update(None, "rekey", &new_pragma)
        .map_err(|e| format!("rekey failed: {e}"))?;
    drop(conn);

    // 4. Re-wrap the new DB key under the launch password so unlock keeps
    //    working. Verify the launch password first by loading the bundle.
    //    (The capsule wraps ride along untouched — the capsule key does not
    //    change with a DB rekey.)
    {
        let mut bundle = load_bundle_anywhere(&state)?
            .ok_or("No key bundle found — the app is not initialized.")?;
        // Prove the launch password is correct (unwrap would yield the OLD key).
        let _ = bundle
            .unwrap_primary(&launch_password)
            .map_err(|_| "Launch password is incorrect.".to_string())?;
        bundle
            .rewrap_for_new_db_key(&new_key, &launch_password)
            .map_err(|e| format!("re-wrap after rekey: {e}"))?;
        bundle
            .save(&state.paths.key_bundle_path)
            .map_err(|e| format!("save key bundle: {e}"))?;
    }

    // 5. Tear down the old-keyed runtime and reboot with the new key.
    {
        let mut tx = state.cmd_tx.lock().await;
        *tx = None; // dropping the sender ends the runtime task
    }
    {
        let mut store_slot = state.store.lock().await;
        *store_slot = None; // drop the old-keyed MemoryStore
    }

    let conn = open_encrypted(&state.paths.db_path, &new_key)
        .map_err(|e| format!("reopen after rekey: {e}"))?;
    boot_runtime(app, state, conn, new_key).await?;
    Ok(())
}

// ===== Encapsulated: capsule sealing ==========================================

use std::sync::atomic::{AtomicBool, Ordering};

/// One seal in flight at a time (turn_done auto-save, pulls, and exit can
/// race each other).
static SEAL_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// Checkpoint the DB, then seal the staging dir into the exe and swap it in
/// (safe while running). Used by the auto-save path and the exit handler.
/// `final_seal = true` additionally stops the backend server processes,
/// closes the DB, and — after a SEALED swap — wipes the staging copy (the
/// exe carries everything; the temp folder keeps only the webview cache).
pub async fn seal_capsule_now(state: &WebState, final_seal: bool) -> Result<String, String> {
    if SEAL_IN_FLIGHT.swap(true, Ordering::SeqCst) {
        return Ok("seal already in flight".into());
    }
    let result = async {
        // 1. Fold the WAL so the sealed memory.db is self-contained.
        {
            let store = state.store.lock().await;
            if let Some(s) = store.as_ref() {
                let s = s.lock().await;
                s.checkpoint_wal().map_err(|e| e.to_string())?;
            }
        }
        // 2. Pick the seal mode. The capsule key in memory decides: present →
        //    v2 sealed (header = the CURRENT bundle from staging — every
        //    bundle-touching command keeps it fresh); absent on an exe that
        //    is already sealed → SKIP (never downgrade, never wipe — staging
        //    is the only state copy until the next unlocked session); absent
        //    on a v1/fresh exe → legacy plaintext seal.
        let exe = current_exe()?;
        let ver = crate::capsule::probe(&exe).map_err(|e| e.to_string())?;
        let key = state.capsule_key.lock().await.clone();
        let mut header_text = String::new();
        let mode = match key.as_ref() {
            Some(k) => {
                let bundle = load_bundle_anywhere(state)?
                    .ok_or("capsule key in memory but no key bundle found")?;
                if !bundle.has_capsule_key() {
                    return Err("key bundle lost its capsule wrap — refusing to seal".into());
                }
                header_text = bundle.to_text().map_err(|e| e.to_string())?;
                crate::capsule::SealMode::Sealed {
                    key: &*k,
                    header: header_text.as_bytes(),
                }
            }
            None => match ver {
                Some(crate::capsule::CapsuleVer::V2) => {
                    return Ok(
                        "sealed capsule is locked (no key this session) — exe kept as-is; \
                         staging retains the state"
                            .into(),
                    );
                }
                _ => crate::capsule::SealMode::Plain,
            },
        };
        // 3. Seal + swap (store lock released — the db file is idle).
        let outcome =
            crate::capsule::seal_and_swap(&exe, &state.paths.data_dir, &mode)
                .map_err(|e| e.to_string())?;
        let mut note = format!(
            "sealed {} entries (+{} bytes, {})",
            outcome.entries,
            outcome.appended,
            if outcome.sealed { "encrypted" } else { "plain v1" }
        );
        if final_seal {
            state.process_mgr.stop_all().await;
            // Release our handles so the wipe can delete what it sealed:
            // the store (open memory.db) and the runtime task (holds the
            // store clone), then the engine's model replicas (mmap'd GGUFs).
            *state.store.lock().await = None;
            *state.cmd_tx.lock().await = None;
            let engine = state.provider.embedded().state();
            for tag in engine.tags().await {
                let _ = engine.remove_model(&tag).await;
            }
            // Wipe the staging copy — only after a SEALED swap that really
            // landed, never in the portable layout (the exe lives in that
            // folder — wiping it would delete the app itself), and never in
            // debug builds (cargo rewrites the dev exe, so its capsule tail
            // cannot be trusted to rehydrate — dev keeps its staging).
            if outcome.sealed && !cfg!(debug_assertions) {
                let swapped =
                    crate::capsule::wait_for_swap(&exe.with_extension("exe.new"), 4000);
                let exe_dir = exe.parent().map(|p| p.to_path_buf());
                if swapped && exe_dir.is_some_and(|d| d != state.paths.data_dir) {
                    note.push_str("; ");
                    note.push_str(&crate::capsule::wipe_staging(&state.paths.data_dir));
                }
            }
        }
        Ok(note)
    }
    .await;
    SEAL_IN_FLIGHT.store(false, Ordering::SeqCst);
    result
}

/// Auto-save — the frontend invokes this on `turn_done` and after model
/// pulls, so the exe always carries at least the previous finished task
/// (power-break / crash safeguard).
#[tauri::command]
pub async fn seal_capsule(state: State<'_, WebState>) -> Result<String, String> {
    seal_capsule_now(&state, false).await
}
