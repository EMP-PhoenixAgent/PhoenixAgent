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
    /// Override the data directory (default ~/.phoenix).
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
    paths.ensure_dirs().context("create data dirs")?;

    let workdir = cli
        .workdir
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let cmd = cli.command.unwrap_or(Cmd::Gui);
    match cmd {
        Cmd::Gui => cmd_gui(&paths, &workdir),
        Cmd::Doctor => rt.block_on(cmd_doctor(&paths, &workdir)),
    }
}

/// Launch the Tauri GUI window.
fn cmd_gui(paths: &Paths, workdir: &std::path::Path) -> anyhow::Result<()> {
    let _guard = phoenix_agent::logging::init(&paths.logs_dir);
    tracing::info!("starting phoenix GUI (workdir={})", workdir.display());

    let cfg = load_config(paths).unwrap_or_default();
    phoenix_agent::web::run(cfg, paths.clone(), workdir.to_path_buf())
        .map_err(anyhow::Error::from)
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
