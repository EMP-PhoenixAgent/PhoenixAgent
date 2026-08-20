//! `ambercore` — the AmberCore CLI.
//!
//! Three subcommands:
//! - `serve`    — run the Ollama-compatible HTTP server (default port [`DEFAULT_PORT`]).
//! - `run`      — one-off chat in the terminal (M0: decode one token; M1: full).
//! - `register` — add a local GGUF file to the catalog under a tag.
//!
//! Each subcommand is wired but stubbed for now; real implementations land with
//! their milestones. The CLI exists so the public API has an executable surface
//! and the subcommand shape is fixed early.

use clap::{Parser, Subcommand};
use ambercore::{catalog::default_models_dir, Catalog, CatalogEntry, DEFAULT_PORT, VERSION};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "ambercore",
    version = VERSION,
    about = "AmberCore — a fully-Rust LLM runner (Candle). Drop-in Ollama-compatible backend.",
    long_about = "AmberCore loads GGUF models and serves them over an Ollama-compatible \
                  HTTP API so the Phoenix Agent can use it as its model backend.\n\n\
                  Default server port: 42069. To point Phoenix at AmberCore, set \
                  `ollama_url = \"http://localhost:42069\"` in Phoenix's config.toml."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the Ollama-compatible HTTP server.
    Serve {
        /// Port to bind (default: 42069).
        #[arg(long, default_value_t = DEFAULT_PORT)]
        port: u16,
        /// Models directory (default: ~/.ambercore/models).
        #[arg(long)]
        models_dir: Option<PathBuf>,
        /// Compute device: cpu, cuda, metal, amd, or auto. `cuda` needs
        /// `--features cuda` + NVIDIA GPU; `metal` needs `--features metal` +
        /// macOS/Apple GPU; `amd` is a stub that errors cleanly; `auto` tries
        /// CUDA → Metal → CPU.
        #[arg(long, value_enum, default_value_t = ambercore::backend::DeviceChoice::Cpu)]
        device: ambercore::backend::DeviceChoice,
        /// Max concurrent model replicas per tag. Default 1 = serialize same-tag
        /// requests (the pre-M5b behavior). Raise to let that many same-tag
        /// requests generate in parallel — at a cost of one full model copy per
        /// replica in RAM/VRAM (8B Q4_K_M ≈5 GB each → keep at 1 on the 8 GB GPU;
        /// on CPU, raise within RAM).
        #[arg(long, default_value_t = 1)]
        max_replicas: usize,
    },
    /// Run a one-off generation against a model.
    Run {
        /// Model tag (e.g. `qwen2.5-coder:7b`) or path to a GGUF file.
        #[arg(long)]
        model: String,
        /// Prompt text. If omitted, the prompt is treated as a user turn in a
        /// chat (Qwen ChatML template); use --raw to send it verbatim.
        #[arg(long)]
        prompt: Option<String>,
        /// Send the prompt verbatim (no chat template). Default: format as a
        /// user turn for instruct models.
        #[arg(long, default_value_t = false)]
        raw: bool,
        /// Sampling temperature (0 = greedy).
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        /// Top-k (0 = disabled).
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Top-p nucleus (>=1.0 = disabled).
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        /// RNG seed for sampling.
        #[arg(long, default_value_t = 299792458)]
        seed: u64,
        /// Maximum tokens to generate. Omit for unlimited (run until EOS).
        #[arg(long)]
        max_tokens: Option<usize>,
        /// Compute device: cpu, cuda, metal, amd, or auto.
        #[arg(long, value_enum, default_value_t = ambercore::backend::DeviceChoice::Cpu)]
        device: ambercore::backend::DeviceChoice,
        /// Models directory (default: ~/.ambercore/models).
        #[arg(long)]
        models_dir: Option<PathBuf>,
    },
    /// Register a local GGUF file under a tag in the catalog.
    Register {
        /// Path to the GGUF file.
        #[arg(long)]
        file: PathBuf,
        /// Tag to register it under (e.g. `qwen2.5-coder:7b`).
        #[arg(long)]
        tag: String,
        /// Optional architecture hint (e.g. `qwen2`, `llama`). Auto-detected if omitted.
        #[arg(long)]
        arch: Option<String>,
        /// Models directory (default: ~/.ambercore/models).
        #[arg(long)]
        models_dir: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    // Initialize structured logging.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Serve {
            port,
            models_dir,
            device,
            max_replicas,
        } => serve(port, models_dir, device, max_replicas)?,
        Command::Run {
            model,
            prompt,
            raw,
            temperature,
            top_k,
            top_p,
            seed,
            max_tokens,
            device,
            models_dir,
        } => run(
            model, prompt, raw, temperature, top_k, top_p, seed, max_tokens, device,
            models_dir,
        )?,
        Command::Register {
            file,
            tag,
            arch,
            models_dir,
        } => register(file, tag, arch, models_dir)?,
    }
    Ok(())
}

/// Resolve the models dir: explicit override, else the default.
fn resolve_models_dir(override_: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(d) = override_ {
        return Ok(d);
    }
    Ok(default_models_dir()?)
}

fn serve(
    port: u16,
    models_dir: Option<PathBuf>,
    device: ambercore::backend::DeviceChoice,
    max_replicas: usize,
) -> anyhow::Result<()> {
    let dir = resolve_models_dir(models_dir)?;
    let catalog = Catalog::load(&dir)?;

    // Resolve the compute backend (cpu / cuda / auto).
    let backend = ambercore::backend::resolve_backend(device)
        .map_err(|e| anyhow::anyhow!("resolve device: {e}"))?;
    tracing::info!(
        "device: {} → backend: {} ({} model(s): [{}])",
        device.as_str(),
        backend.name(),
        catalog.tags().len(),
        catalog.tags().join(", ")
    );
    eprintln!(
        "ambercore serve: port={} device={} ({} model(s) registered)",
        port,
        backend.name(),
        catalog.tags().len()
    );

    let state = ambercore::server::ServerState::new(catalog, backend, max_replicas);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(ambercore::server::serve(port, state))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run(
    model: String,
    prompt: Option<String>,
    raw: bool,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    max_tokens: Option<usize>,
    device: ambercore::backend::DeviceChoice,
    models_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    use ambercore::{
        backend::resolve_backend,
        model::{build as build_model, LoadedModel},
        pipeline::{Pipeline, SampleParams, StopCondition},
        tokenizer::{format_qwen_chatml, ChatTurn, Role, TokenizerWrapper},
    };
    use std::io::Write;
    use std::time::Instant;

    let dir = resolve_models_dir(models_dir)?;
    let catalog = Catalog::load(&dir)?;

    let entry = catalog.get(&model).cloned().or_else(|| {
        let p = PathBuf::from(&model);
        if p.is_file() {
            Some(CatalogEntry {
                tag: model.clone(),
                file: p.to_string_lossy().into_owned(),
                arch: None,
            })
        } else {
            None
        }
    });

    let Some(entry) = entry else {
        anyhow::bail!(
            "model '{}' not found in catalog and not a file path. Known tags: [{}]",
            model,
            catalog.tags().join(", ")
        );
    };

    let path = catalog.resolve_path(&entry);
    let user_prompt = prompt.unwrap_or_else(|| {
        eprintln!("(no --prompt given; using a default for testing)");
        "Hello!".to_string()
    });

    // Report SIMD/CPU capabilities (useful for perf reasoning).
    eprintln!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c(),
    );

    // 1. Load the GGUF.
    let t0 = Instant::now();
    let mut loaded = LoadedModel::load(&path)
        .map_err(|e| anyhow::anyhow!("load GGUF {}: {e}", path.display()))?;
    eprintln!(
        "loaded GGUF: arch={} name={:?} (in {:.2}s)",
        loaded.arch,
        loaded.name,
        t0.elapsed().as_secs_f32()
    );

    // 2. Resolve the backend (cpu / cuda / auto) + build the model.
    let backend = resolve_backend(device).map_err(|e| anyhow::anyhow!("resolve device: {e}"))?;
    eprintln!("device backend: {}", backend.name());
    let device = backend
        .device()
        .map_err(|e| anyhow::anyhow!("backend device: {e}"))?;
    // Capture the arch + context length before build() consumes the GGUF content.
    let arch_owned = loaded.arch.clone();
    let context_length = loaded
        .meta_str(&format!("{}.context_length", loaded.arch))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4096);
    let t1 = Instant::now();
    let mut model = build_model(&mut loaded, &device)
        .map_err(|e| anyhow::anyhow!("build model: {e}"))?;
    eprintln!(
        "built {} model (ctx {} tokens, in {:.2}s)",
        model.arch(),
        context_length,
        t1.elapsed().as_secs_f32()
    );
    let _ = arch_owned;

    // 3. Load the tokenizer + resolve stop tokens.
    let tokenizer = TokenizerWrapper::load_next_to(&path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let prompt_text = if raw {
        user_prompt.clone()
    } else {
        format_qwen_chatml(&[ChatTurn { role: Role::User, content: user_prompt.clone() }], None)
    };

    // Qwen instruct models end the assistant turn with <|im_end|> or <|endoftext|>.
    let mut stop_tokens = Vec::new();
    if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
        stop_tokens.push(id);
    }
    if let Some(id) = tokenizer.token_to_id("<|endoftext|>") {
        stop_tokens.push(id);
    }

    let params = SampleParams {
        temperature,
        top_k,
        top_p,
        seed,
    };
    let stop = StopCondition {
        max_tokens,
        stop_tokens,
    };

    // 4. Stream the generation.
    let mut pipeline = Pipeline {
        model: model.as_mut(),
        tokenizer: &tokenizer,
        device: &device,
        context_length,
    };

    eprintln!("--- generation ---");
    let (stats, full) = pipeline
        .generate(&prompt_text, &params, &stop, |delta| {
            print!("{delta}");
            let _ = std::io::stdout().flush();
        })
        .map_err(|e| anyhow::anyhow!("generate: {e}"))?;
    println!();
    println!("--- end ({}, {:.2} tok/s) ---", stats.output_tokens, stats.tokens_per_sec());

    // Keep full text referenced so it's not optimized away in debug.
    let _ = full;
    Ok(())
}

fn register(
    file: PathBuf,
    tag: String,
    arch: Option<String>,
    models_dir: Option<PathBuf>,
) -> anyhow::Result<()> {
    let dir = resolve_models_dir(models_dir)?;
    std::fs::create_dir_all(&dir)?;

    // The catalog stores `file` relative to the models dir when possible.
    let abs = file.canonicalize()?;
    let rel = abs
        .strip_prefix(&dir.canonicalize().unwrap_or(dir.clone()))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| abs.to_string_lossy().into_owned());

    let entry = CatalogEntry {
        tag,
        file: rel,
        arch,
    };

    let mut catalog = Catalog::load(&dir)?;
    catalog.register(entry)?;
    tracing::info!("registered model; catalog now: [{}]", catalog.tags().join(", "));
    Ok(())
}
