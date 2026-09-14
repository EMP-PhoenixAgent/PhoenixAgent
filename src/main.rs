//! Phoenix Agent entry point.
//!
//! Usage:
//! - `phoenix`          Launch the GUI window (default).
//! - `phoenix doctor`   Diagnose Ollama, DB, and toolchain.

use std::path::PathBuf;

use anyhow::Context as _;
use clap::{Parser, Subcommand};

use phoenix_agent::config::{load_config, Config, Paths};
use phoenix_agent::model::ollama::OllamaProvider;
use phoenix_agent::model::ModelProvider as _;

#[derive(Parser, Debug)]
#[command(
    name = "phoenix",
    version,
    about = "Phoenix Agent — a fully-local, encrypted, autonomous coding agent"
)]
struct Cli {
    /// Override the data directory (default: the installation folder — the
    /// directory the app is running from).
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,

    /// Working directory / project root (default: current directory).
    #[arg(long, global = true)]
    workdir: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Diagnose the environment: Ollama, DB, toolchain.
    Doctor,
    /// Launch the Tauri GUI window (default action).
    Gui,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let data_dir = cli
        .data_dir
        .clone()
        .unwrap_or_else(Paths::default_data_dir);
    let paths = Paths::new(data_dir);
    paths.ensure_dirs().context("prepare data dirs")?;
    // One-time migration from the pre-portable layout (~/.phoenix +
    // ~/.ambercore) into the install-folder data root. No-op on fresh
    // installs and after the first run.
    let legacy_home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    phoenix_agent::config::migrate_legacy_data(&paths, &legacy_home);

    let workdir = cli
        .workdir
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let cmd = cli.command.unwrap_or(Cmd::Gui);
    // Encapsulated boot: clean up any leftover swap artifact, resolve/create
    // the models folder BESIDE the exe, and hydrate the data dir from the
    // exe's state capsule. A v1 (plaintext) capsule hydrates here as before
    // (its models/ entries go to the exe-side models folder — the legacy
    // layout kept them in the capsule); a v2 (sealed) capsule CANNOT hydrate
    // — the tail is ciphertext until the launch password arrives, so its
    // hydration happens in the unlock command, behind the gate. Models are
    // NEVER sealed anymore; the temp staging area holds only private state +
    // the webview cache. (Logging is not up yet — failures go to stderr.)
    if matches!(cmd, Cmd::Gui) {
        if let Ok(exe) = std::env::current_exe() {
            phoenix_agent::capsule::sweep_old(&exe);
            let models_home = phoenix_agent::config::encaps_models_dir(&paths.data_dir);
            match phoenix_agent::capsule::probe(&exe) {
                Ok(Some(phoenix_agent::capsule::CapsuleVer::V1)) => {
                    if let Ok(Some(cap)) = phoenix_agent::capsule::Capsule::open(&exe, None) {
                        for e in &cap.entries {
                            let root = if e.name.starts_with("models/") {
                                &models_home
                            } else {
                                &paths.data_dir
                            };
                            let out = root.join(&e.name);
                            if !out.exists() {
                                if let Err(err) = cap.extract(&exe, &e.name, &out) {
                                    eprintln!("capsule: hydration failed for {}: {err}", e.name);
                                }
                            }
                        }
                    }
                }
                Ok(Some(phoenix_agent::capsule::CapsuleVer::V2)) => {
                    // Sealed — hydration is deferred to `unlock`.
                }
                _ => {}
            }
            if std::env::var_os("WEBVIEW2_USER_DATA_FOLDER").is_none() {
                let staging = phoenix_agent::config::Paths::encaps_staging_dir();
                std::env::set_var("WEBVIEW2_USER_DATA_FOLDER", staging.join("webview"));
            }
        }
    }

    // Pin our rendering to the iGPU BEFORE any thread spawns (env-var safety)
    // and before WebView2 launches (it reads the var at creation).
    let igpu_note = if matches!(cmd, Cmd::Gui) {
        pin_rendering_to_igpu()
    } else {
        None
    };
    match cmd {
        Cmd::Gui => cmd_gui(&paths, &workdir, igpu_note),
        Cmd::Doctor => rt.block_on(cmd_doctor(&paths, &workdir)),
    }
}

/// Launch the Tauri GUI window.
fn cmd_gui(paths: &Paths, workdir: &std::path::Path, igpu_note: Option<String>) -> anyhow::Result<()> {
    let _guard = phoenix_agent::logging::init(&paths.logs_dir);
    tracing::info!("starting phoenix GUI (workdir={})", workdir.display());
    if let Some(note) = igpu_note {
        tracing::info!("{}", note);
    }
    // One-time move: models that the pre-Phase-C build kept in the temp
    // staging area (`<staging>/models`) belong beside the exe now.
    phoenix_agent::config::migrate_staging_models(
        &paths.data_dir.join("models"),
        &phoenix_agent::config::encaps_models_dir(&paths.data_dir),
    );

    let cfg = load_config(paths).unwrap_or_default();
    phoenix_agent::web::run(cfg, paths.clone(), workdir.to_path_buf())
        .map_err(anyhow::Error::from)
}

/// Pin Phoenix's own rendering (the WebView2 UI) to the integrated GPU when
/// one is present, keeping the discrete GPU's VRAM free for the model — the
/// chat UI needs nothing from the RTX, the model needs all of it. Returns a
/// log line when pinned; `None` when no iGPU or not on Windows.
///
/// Two mechanisms, both idempotent:
/// 1. `--prefer-integrated-gpu` through `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS`
///    — Chromium's documented switch; must be set before the WebView2 spawns.
///    CUDA is NOT affected: it enumerates NVIDIA devices regardless of the
///    process's DXGI adapter preference, so the model stays on the dGPU.
/// 2. The per-exe Windows GPU preference (`HKCU\Software\Microsoft\DirectX\
///    UserGpuPreferences` → `"GpuPreference=1;"` under the exe's path) — the
///    persistent OS-level record ("always use the power-saving GPU for this
///    app"), covering any DirectX usage in our own process too.
fn pin_rendering_to_igpu() -> Option<String> {
    #[cfg(windows)]
    {
        let igpu = ambercore::backend::detect_secondary_gpu()?;
        // 1. Chromium switch (merged with any pre-set arguments).
        let existing = std::env::var("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS").unwrap_or_default();
        if !existing.contains("prefer-integrated-gpu") {
            let merged = if existing.trim().is_empty() {
                "--prefer-integrated-gpu".to_string()
            } else {
                format!("{existing} --prefer-integrated-gpu")
            };
            // Sound: called before the tokio runtime spawns worker threads.
            std::env::set_var("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", merged);
        }
        // 2. Persistent per-exe preference (HKCU needs no elevation; reg add
        //    is idempotent with /f).
        if let Ok(exe) = std::env::current_exe() {
            let _ = std::process::Command::new("reg")
                .args([
                    "add",
                    r"HKCU\Software\Microsoft\DirectX\UserGpuPreferences",
                    "/v",
                ])
                .arg(exe.as_os_str())
                .args(["/t", "REG_SZ", "/d", "GpuPreference=1;", "/f"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
        }
        Some(format!(
            "iGPU detected ({igpu}) — Phoenix rendering pinned to it; the discrete \
             GPU's VRAM stays reserved for the model (WebView2 switch applies from this \
             launch; the saved per-app preference persists for future launches)"
        ))
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Diagnostic checks (no passphrase prompt — read-only where possible).
async fn cmd_doctor(paths: &Paths, workdir: &std::path::Path) -> anyhow::Result<()> {
    println!("Phoenix Agent — diagnostics\n");

    let cfg = load_config(paths).unwrap_or_else(|e| {
        println!("[warn] config: {e}; using defaults");
        Config::default()
    });
    println!("[config] data_dir   : {}", paths.data_dir.display());
    println!("[config] config.toml: {}", paths.config_path.display());
    println!("[config] db         : {}", paths.db_path.display());
    println!("[config] model      : {}", cfg.model);
    println!("[config] ollama_url : {}", cfg.ollama_url);

    // Ollama reachability + model presence.
    let provider = OllamaProvider::new(&cfg.ollama_url);
    match provider.list_models().await {
        Ok(models) => {
            let has = models.iter().any(|m| m == &cfg.model);
            println!(
                "[ollama] reachable · {} model(s) local · default {} present",
                models.len(),
                if has { "IS" } else { "is NOT" }
            );
            if !has {
                println!("         -> run `ollama pull {}`", cfg.model);
            }
        }
        Err(e) => {
            println!("[ollama] NOT reachable: {e}");
            println!("         -> start with `ollama serve`");
        }
    }

    // DB existence (don't prompt for passphrase in this mode).
    if paths.db_path.exists() {
        println!("[db] present (encrypted). Use the GUI to unlock.");
    } else {
        println!("[db] not initialized — launch the GUI (`phoenix`) to set up.");
    }

    println!("\n[workdir] {}", workdir.display());
    println!("\nDone.");
    Ok(())
}
