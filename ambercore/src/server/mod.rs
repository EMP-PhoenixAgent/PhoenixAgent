//! HTTP server — the Ollama-compatible drop-in for Phoenix Agent.
//!
//! axum app exposing exactly the two endpoints Phoenix touches:
//!
//! - `GET  /api/tags` — model listing (also Phoenix's health probe)
//! - `POST /api/chat` — streaming chat (NDJSON, one JSON object per line)
//!
//! Wire types live in [`protocol`] and match the audited Phoenix contract
//! exactly. Handlers live in [`tags`] and [`chat`].
//!
//! ## Concurrency model
//!
//! Loading a GGUF + building the model is expensive (seconds + GiB of RAM), so
//! [`ServerState`] keeps a [`ReplicaPool`] per tag holding up to `max_replicas`
//! built models. Generation is CPU/GPU-bound and runs on
//! `tokio::task::spawn_blocking` so it doesn't stall the async runtime. A candle
//! quantized model is single-threaded (`&mut self` with an internal KV cache),
//! so concurrent same-tag requests each need their own replica: the pool hands
//! out free replicas, grows lazily up to the cap, and FIFO-queues beyond it. At
//! `max_replicas = 1` this reduces to the old per-tag serialization. (True
//! token-level batching — one matmul across sequences — is blocked on candle's
//! single-sequence quantized KV; see ACRoad.md §7 M5b.)

pub mod chat;
pub mod protocol;
pub mod stats;
pub mod tags;
pub mod telemetry;
pub mod tools;

use crate::backend::Backend;
use crate::catalog::{Catalog, CatalogEntry};
use crate::error::{Error, Result};
use crate::model::{build as build_model, DynModel, LoadedModel};
use crate::tokenizer::TokenizerWrapper;
use axum::routing::{get, post};
use axum::Router;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Engine-side lifecycle events, fanned out to any subscriber (the embedding
/// host taps this for its session log; the standalone HTTP server simply has
/// none). The reason it exists: **silent recoveries** — above all the OOM
/// CPU fallback — never fail a request, so the host has no other way to learn
/// they happened.
#[derive(Debug, Clone)]
pub enum EngineEvent {
    /// A model build ran out of VRAM on the GPU and was rebuilt on the CPU.
    /// The requesting turn still succeeds — slower.
    OomFallback { tag: String },
    /// A model replica finished building and entered its pool.
    ModelLoaded { tag: String, device: String, dur_ms: u64 },
    /// A model's replicas were unloaded (mmaps released).
    ModelUnloaded { tag: String },
    /// A pre-flight VRAM verdict worth surfacing (refusal / spill warning).
    VramWarning { message: String },
}

/// End-of-turn token strings per architecture family. Tokens absent from the
/// loaded tokenizer are skipped, and the GGUF's `<arch>.eos_token_id` is added
/// separately (by id, so models with unusual EOS strings still stop).
fn arch_stop_markers(arch: &str) -> &'static [&'static str] {
    match arch {
        "gemma" | "gemma2" | "gemma3" | "gemma4" => &["<end_of_turn>", "<|endoftext|>"],
        "phi2" => &["<|endoftext|>"],
        "phi3" => &["<|end|>", "<|endoftext|>"],
        "glm4" => &["<|user|>", "<|observation|>", "<|endoftext|>"],
        "mixtral" => &["</s>", "<|end_of_text|>"],
        "llama" | "nemotron" => {
            // `<|end_response|>` only exists on Nemotron-Hybrid/Super builds.
            &["<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>", "<|end_response|>"]
        }
        "granite" | "granitemoe" | "granite_swa" => {
            // `<|end_of_planning|>` only exists on Granite-4 thinking models.
            &["<|end_of_text|>", "<|end_of_planning|>", "<|endoftext|>"]
        }
        // The deepseek2 arch covers DeepSeek V2/V3/R1 (full-width markers) and
        // Kimi K2 (im_* markup); Kimi's [EOS] rides the GGUF eos_token_id.
        "deepseek2" | "deepseek32" | "kimi_k2" => {
            &["<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>", "<|im_end|>"]
        }
        "minimax-m2" => &["[e~[", "]~b]user", "]~b]tool", "<|end_of_text|>", "<|im_end|>"],
        // qwen2/qwen2_v2/qwen3/qwen3moe/starcoder2/internlm2/lfm2 + default
        _ => &["<|im_end|>", "<|endoftext|>", "<|end_of_text|>"],
    }
}

/// A built model + its tokenizer, ready to serve generation requests.
///
/// Must be `Send` so it can live behind an `Arc<std::sync::Mutex<...>>` shared
/// across the thread pool. candle's quantized tensors are `Send` on CPU.
pub struct LoadedEntry {
    pub model: Box<dyn DynModel>,
    pub tokenizer: TokenizerWrapper,
    pub arch: String,
    pub stop_tokens: Vec<u32>,
    /// The device the model lives on (CPU or Cuda). The pipeline places input
    /// tensors on this device so they match the model's weights.
    pub device: candle_core::Device,
    /// The model's trained context length (max prompt tokens). Surfaced from
    /// the GGUF `<arch>.context_length` metadata. Used to warn (not block) when
    /// a prompt exceeds it.
    pub context_length: usize,
}

/// Shared server state. Cheaply cloneable (`Arc`).
#[derive(Clone)]
pub struct ServerState {
    inner: Arc<Inner>,
}

struct Inner {
    /// The catalog maps tags → GGUF file paths. Async-mutex so `/api/tags` can
    /// read it without blocking the runtime.
    catalog: tokio::sync::Mutex<Catalog>,
    /// One replica pool per tag, lazily created on first request. Each pool owns
    /// up to `max_replicas` built models and hands them out fairly (M5b).
    pools: tokio::sync::Mutex<HashMap<String, Arc<ReplicaPool<Replica>>>>,
    /// Max built-model replicas per tag. 1 = serialize per tag (the pre-M5b
    /// behavior); >1 = that many concurrent generations per tag, memory allowing
    /// (each replica = full model size).
    max_replicas: usize,
    /// The compute backend (CPU or Cuda).
    backend: Box<dyn Backend>,
    /// Last measured generation throughput (tokens/sec), updated at the end of
    /// each `/api/chat` generation. Surfaced via `GET /api/stats` for the UI's
    /// health bar. `None` until the first generation completes.
    last_tokens_per_sec: tokio::sync::Mutex<Option<f64>>,
    /// Hardware snapshot captured once at startup — CPU model/cores/RAM/OS +
    /// (cuda) GPU. Cached in an `Arc` so telemetry push tasks clone it cheaply.
    hardware: Arc<telemetry::Hardware>,
    /// Low-VRAM refusal/warning from the most recent model build attempt
    /// (None = fit or unknown). Stored so the UI can surface it after a
    /// warm-up.
    last_vram_warning: std::sync::Mutex<Option<String>>,
    /// Positive "contained in VRAM" line from the most recent successful
    /// pre-flight check (chat verdict for the fit case).
    last_vram_info: std::sync::Mutex<Option<String>>,
    /// Fan-out for [`EngineEvent`]s — the host's window onto silent
    /// recoveries. No subscribers = sends are free no-ops.
    events: broadcast::Sender<EngineEvent>,
}

impl ServerState {
    pub fn new(catalog: Catalog, backend: Box<dyn Backend>, max_replicas: usize) -> Self {
        let backend_name = backend.name().to_string();
        let hardware = Arc::new(telemetry::hardware_snapshot(&backend_name));
        let max_replicas = max_replicas.max(1);
        tracing::info!(
            backend = %backend_name,
            max_replicas,
            cpu = ?hardware.cpu,
            cores = ?hardware.cpu_cores,
            ram_mb = ?hardware.ram_total_mb,
            "telemetry: hardware snapshot captured"
        );
        Self {
            inner: Arc::new(Inner {
                catalog: tokio::sync::Mutex::new(catalog),
                pools: tokio::sync::Mutex::new(HashMap::new()),
                max_replicas,
                backend,
                last_tokens_per_sec: tokio::sync::Mutex::new(None),
                hardware,
                last_vram_warning: std::sync::Mutex::new(None),
                last_vram_info: std::sync::Mutex::new(None),
                events: broadcast::channel(128).0,
            }),
        }
    }

    /// Subscribe to the engine's lifecycle events (see [`EngineEvent`]).
    /// Receivers that fall behind lose old events (broadcast semantics) —
    /// fine for logs, never for control flow.
    pub fn subscribe(&self) -> broadcast::Receiver<EngineEvent> {
        self.inner.events.subscribe()
    }

    /// Fan one event out to every subscriber (no-op when nobody listens).
    fn emit(&self, event: EngineEvent) {
        let _ = self.inner.events.send(event);
    }

    /// The low-VRAM refusal/warning from the most recent model build attempt,
    /// if any (None = the model fit, or the check couldn't run).
    pub fn last_vram_warning(&self) -> Option<String> {
        self.inner.last_vram_warning.lock().unwrap().clone()
    }

    /// The "contained in VRAM" line from the most recent pre-flight check
    /// that passed, if any.
    pub fn last_vram_info(&self) -> Option<String> {
        self.inner.last_vram_info.lock().unwrap().clone()
    }

    /// The cached hardware snapshot (for telemetry push + `/api/telemetry/status`).
    pub fn hardware(&self) -> Arc<telemetry::Hardware> {
        self.inner.hardware.clone()
    }

    /// The backend's display name (e.g. "cpu" or "cuda").
    pub fn backend_name(&self) -> &str {
        self.inner.backend.name()
    }

    /// Hardware check-up for the UI: the boot-time CPU/RAM/OS snapshot merged
    /// with the active backend name and a live GPU reading (name + VRAM) when
    /// a GPU backend is in use.
    pub fn hardware_status(&self) -> telemetry::HardwareStatus {
        let hw = &self.inner.hardware;
        telemetry::HardwareStatus {
            backend: self.inner.backend.name().to_string(),
            cpu: hw.cpu.clone(),
            cpu_cores: hw.cpu_cores,
            ram_total_mb: hw.ram_total_mb,
            os: hw.os.clone(),
            gpu: self.inner.backend.gpu_info(),
            vram_warning: self.last_vram_warning(),
        }
    }

    /// Record the throughput from the most recent generation (called at the end
    /// of each `/api/chat` request). Powers `GET /api/stats`.
    pub async fn record_tokens_per_sec(&self, tps: f64) {
        *self.inner.last_tokens_per_sec.lock().await = Some(tps);
    }

    /// The last measured throughput, if any generation has completed.
    pub async fn last_tokens_per_sec(&self) -> Option<f64> {
        *self.inner.last_tokens_per_sec.lock().await
    }

    /// The list of known tags (for `/api/tags`).
    pub async fn tags(&self) -> Vec<String> {
        self.inner.catalog.lock().await.tags()
    }

    /// Get-or-create the replica pool for a tag. Pools start empty; replicas are
    /// built lazily by [`Self::acquire_replica`].
    async fn pool_for(&self, tag: &str) -> Result<Arc<ReplicaPool<Replica>>> {
        let max = self.inner.max_replicas;
        // Fast path: pool already exists for this tag.
        if let Some(p) = self.inner.pools.lock().await.get(tag).cloned() {
            return Ok(p);
        }
        // Create a fresh empty pool (lazy — no model built yet).
        let p = Arc::new(ReplicaPool::<Replica>::new(max));
        let inserted = self
            .inner
            .pools
            .lock()
            .await
            .entry(tag.to_string())
            .or_insert_with(|| p.clone())
            .clone();
        Ok(inserted)
    }

    /// The canonical catalog spelling of `tag` (see [`Catalog::resolve`]) —
    /// bare-stem and `:latest` spellings of the same model collapse onto one
    /// entry, and one replica pool.
    async fn canonical_tag(&self, tag: &str) -> Result<String> {
        self.inner
            .catalog
            .lock()
            .await
            .resolve(tag)
            .map(|(_, canonical)| canonical)
            .ok_or_else(|| Error::NotFound(format!("model tag '{tag}' not in catalog")))
    }

    /// Resolve a tag to its GGUF and build a fresh [`LoadedEntry`] (load GGUF,
    /// build the model, load the tokenizer, resolve stop tokens). The candle
    /// parsing runs on a blocking thread. Factored out so the first replica and
    /// later pool-grown replicas share one build path.
    async fn build_loaded_entry(&self, tag: &str) -> Result<LoadedEntry> {
        let (gguf_path, canonical_tag, _arch_hint) = {
            let cat = self.inner.catalog.lock().await;
            let (entry, canonical) = cat.resolve(tag).ok_or_else(|| {
                let known = cat.tags();
                let hint = if known.is_empty() {
                    "the catalog is empty — pull or register a model first".to_string()
                } else {
                    let sample: Vec<&str> =
                        known.iter().take(4).map(|s| s.as_str()).collect();
                    format!("known tags: {}", sample.join(", "))
                };
                Error::NotFound(format!("model tag '{tag}' not in catalog ({hint})"))
            })?;
            (cat.resolve_path(entry), canonical, entry.arch.clone())
        };
        tracing::info!(tag, "building model replica from {}", gguf_path.display());
        let build_started = std::time::Instant::now();
        let device = self.inner.backend.device()?;
        let backend_name = self.inner.backend.name().to_string();
        let backend_name_outer = backend_name.clone();
        let tag_owned = canonical_tag.clone();
        // Pre-flight VRAM check, BEFORE the build takes its share (blocking
        // thread — nvidia-smi subprocess). Hard refusal when the model would
        // spill; see backend::check_vram for the estimate + rationale.
        let check_path = gguf_path.clone();
        let check = tokio::task::spawn_blocking(move || crate::backend::check_vram(&check_path))
            .await
            .map_err(|e| Error::Server(format!("vram check join: {e}")))?;
        {
            let mut warning_slot = self.inner.last_vram_warning.lock().unwrap();
            let mut info_slot = self.inner.last_vram_info.lock().unwrap();
            match check.fits {
                Some(false) => {
                    *warning_slot = check.message.clone();
                    *info_slot = None;
                }
                Some(true) => {
                    *warning_slot = None;
                    *info_slot = check.message.clone();
                }
                None => {
                    *warning_slot = None;
                    *info_slot = None;
                }
            }
        }
        if let Some(msg) = &check.message {
            match check.fits {
                Some(true) => tracing::info!("{}", msg),
                _ => tracing::warn!("{}", msg),
            }
        }
        if check.fits == Some(false) {
            self.emit(EngineEvent::VramWarning { message: check.message.clone().unwrap_or_default() });
            return Err(Error::Backend(
                check
                    .message
                    .unwrap_or_else(|| "model exceeds free VRAM; refusing to load".into()),
            ));
        }
        let entry: LoadedEntry =
        tokio::task::spawn_blocking({
            // Cloned into the blocking closure so the silent OOM recovery can
            // report itself (the host's log would otherwise never see it).
            let events = self.inner.events.clone();
            move || -> Result<LoadedEntry> {
            let mut loaded = LoadedModel::load(&gguf_path)?;
            let arch = loaded.arch.clone();
            // Surface the trained context length from the GGUF metadata
            // (`<arch>.context_length`). Falls back to a large default if absent.
            let context_length = loaded
                .meta_str(&format!("{arch}.context_length"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4096);
            // RWKV world models embed their vocab in the GGUF and it is
            // AUTHORITATIVE — the HF tokenizer.json ports segment text
            // differently (foreign ids → word-salad output). Everyone else
            // loads the sibling tokenizer.json as before.
            let (tokenizer, eos_meta, eot_meta) = {
                let metadata = &loaded
                    .content
                    .as_ref()
                    .ok_or_else(|| Error::Model("GGUF content unavailable".into()))?
                    .metadata;
                let u32_of = |k: &str| {
                    metadata.get(k).and_then(|v| match v {
                        candle_core::quantized::gguf_file::Value::U32(n) => Some(*n),
                        _ => None,
                    })
                };
                let eot = u32_of("tokenizer.ggml.eot_token_id");
                let eos = u32_of("tokenizer.ggml.eos_token_id");
                if TokenizerWrapper::is_rwkv_world(metadata) {
                    let t = TokenizerWrapper::from_rwkv_metadata(metadata)?;
                    tracing::info!(
                        "using the GGUF's embedded RWKV world tokenizer ({} tokens)",
                        t.vocab_size
                    );
                    (t, eos, eot)
                } else {
                    (TokenizerWrapper::load_next_to(&gguf_path)?, eos, eot)
                }
            };
            let eos_meta = loaded
                .meta_str(&format!("{arch}.eos_token_id"))
                .and_then(|s| s.parse::<u32>().ok())
                .or(eos_meta);
            // The pre-flight estimate said it fits, but the driver decides at
            // allocation time — on a CUDA out-of-memory, fall back to CPU for
            // THIS model (slower, but it runs) instead of failing the turn.
            let mut device = device;
            let model = match build_model(&mut loaded, &device) {
                Ok(m) => m,
                Err(e) => {
                    let raw = e.to_string();
                    if crate::backend::is_cuda_oom(&raw)
                        && !matches!(device, candle_core::Device::Cpu)
                    {
                        tracing::warn!(
                            tag = %tag_owned,
                            "GPU build ran out of VRAM — retrying on CPU"
                        );
                        let _ = events.send(EngineEvent::OomFallback { tag: tag_owned.clone() });
                        device = candle_core::Device::Cpu;
                        build_model(&mut loaded, &device).map_err(|e| {
                            crate::error::Error::Model(
                                crate::backend::translate_cuda_error(&e.to_string()),
                            )
                        })?
                    } else {
                        // Defense in depth: the backend constructor already warms
                        // up a kernel and rejects driver/toolkit skews with the
                        // translated message — this catches any CUDA error that
                        // still escapes a model build (e.g. arch-specific kernel
                        // gaps).
                        return Err(crate::error::Error::Model(
                            crate::backend::translate_cuda_error(&raw),
                        ));
                    }
                }
            };
            // Stop tokens: the GGUF's own `<arch>.eos_token_id` plus the
            // per-architecture end-of-turn markers (absent tokens are skipped,
            // so a family's list can include markers only some models carry).
            let mut stop_tokens = Vec::new();
            if let Some(id) = eos_meta {
                stop_tokens.push(id);
            }
            if let Some(id) = eot_meta {
                if !stop_tokens.contains(&id) {
                    stop_tokens.push(id);
                }
            }
            for marker in arch_stop_markers(&arch) {
                if let Some(id) = tokenizer.token_to_id(marker) {
                    if !stop_tokens.contains(&id) {
                        stop_tokens.push(id);
                    }
                }
            }
            tracing::info!(tag = %tag_owned, arch = %arch, backend = %backend_name, context_length, "model replica ready");
            Ok(LoadedEntry {
                model,
                tokenizer,
                arch,
                stop_tokens,
                device: device.clone(),
                context_length,
            })
            }
        })
        .await
        .map_err(|e| Error::Server(format!("load task join: {e}")))??;
        let device_name = match &entry.device {
            candle_core::Device::Cpu => "cpu".to_string(),
            candle_core::Device::Cuda(_) => "cuda".to_string(),
            _ => backend_name_outer,
        };
        self.emit(EngineEvent::ModelLoaded {
            tag: canonical_tag,
            device: device_name,
            dur_ms: build_started.elapsed().as_millis() as u64,
        });
        Ok(entry)
    }

    /// Acquire a model replica for `tag` for the duration of one generation.
    ///
    /// - If a free replica exists, it is handed out immediately.
    /// - Else if the pool can still grow (below `max_replicas`), a new replica is
    ///   built lazily and handed out.
    /// - Else the request queues FIFO and resumes when a replica is released.
    ///
    /// The returned [`ReplicaHandle`] returns the replica to the pool on drop —
    /// so moving it into the generation task and letting it drop at the end is
    /// sufficient (release happens after the surrounding locks/guards drop, so
    /// the replica's mutex is already unlocked by then).
    pub async fn acquire_replica(&self, tag: &str) -> Result<ReplicaHandle<Replica>> {
        // Canonical spelling first so `model` and `model:latest` share one pool.
        let canonical = self.canonical_tag(tag).await?;
        let pool = self.pool_for(&canonical).await?;
        loop {
            match pool.acquire() {
                AcquireOutcome::Ready(r) => {
                    return Ok(ReplicaHandle { pool: pool.clone(), replica: Some(r) });
                }
                AcquireOutcome::Build => {
                    // Grow the pool. Build OUTSIDE the pool lock (expensive).
                    let entry = match self.build_loaded_entry(&canonical).await {
                        Ok(e) => e,
                        Err(e) => {
                            pool.build_failed();
                            return Err(e);
                        }
                    };
                    let replica: Replica = Arc::new(std::sync::Mutex::new(entry));
                    let r = pool.adopt(replica);
                    return Ok(ReplicaHandle { pool: pool.clone(), replica: Some(r) });
                }
                AcquireOutcome::Wait(rx) => {
                    // At capacity; wait for a released replica.
                    let r = rx
                        .await
                        .map_err(|_| Error::Server("replica waiter dropped".into()))?;
                    return Ok(ReplicaHandle { pool: pool.clone(), replica: Some(r) });
                }
            }
        }
    }

    /// Reload the catalog from disk (for a future refresh endpoint).
    pub async fn reload_catalog(&self, models_dir: PathBuf) -> Result<()> {
        let new_cat = Catalog::load(&models_dir)?;
        *self.inner.catalog.lock().await = new_cat;
        Ok(())
    }

    /// Register a model entry in the catalog (persists to the models dir's
    /// `manifest.json` and updates the in-memory map). Used by Phoenix's
    /// in-process pull flow — no `ambercore register` subprocess needed.
    pub async fn register_entry(&self, entry: CatalogEntry) -> Result<()> {
        self.inner.catalog.lock().await.register(entry)
    }

    /// Remove a model from the catalog (persisting to `manifest.json`) and drop
    /// its replica pool so any loaded replica — and its GGUF mmap — is released
    /// before the caller deletes the file (Windows refuses to delete
    /// memory-mapped files). Called by Phoenix's delete-model UI action; file
    /// deletion stays with the caller, who knows the layout. A generation
    /// holding a replica handle keeps its `Arc` until it finishes — the pool
    /// entry is gone either way, so no new generation can acquire it.
    pub async fn remove_model(&self, tag: &str) -> Result<()> {
        let stem = tag.split(':').next().unwrap_or(tag).to_string();
        let canonical = {
            let cat = self.inner.catalog.lock().await;
            cat.resolve(tag).map(|(_, c)| c).unwrap_or_else(|| tag.to_string())
        };
        {
            // Drop every spelling of the same model (canonical + requested +
            // the stem / `:latest` twins) — only the canonical one is
            // persisted-away from manifest.json; the rest are scan aliases.
            let mut cat = self.inner.catalog.lock().await;
            cat.remove(&canonical)?;
            if canonical != tag {
                let _ = cat.remove(tag);
            }
            let with_latest = format!("{stem}:latest");
            if with_latest != canonical && with_latest != tag {
                let _ = cat.remove(&with_latest);
            }
            if stem != canonical && stem != tag {
                let _ = cat.remove(&stem);
            }
        }
        let mut pools = self.inner.pools.lock().await;
        pools.remove(&canonical);
        if canonical != tag {
            pools.remove(tag);
        }
        Ok(())
    }

    /// Unload a model's pooled replicas WITHOUT touching the catalog or
    /// `manifest.json` — the registration must survive restarts. Drops the
    /// pool (releasing GGUF mmaps once in-flight generations finish); a later
    /// acquire rebuilds it from the catalog. Phoenix's EXIT path uses this to
    /// release file handles: calling [`Self::remove_model`] there instead
    /// deregistered every model from the manifest on each exit, so the next
    /// launch only saw scan-derived tags and the active model stopped
    /// resolving.
    pub async fn unload_model(&self, tag: &str) -> Result<()> {
        let stem = tag.split(':').next().unwrap_or(tag).to_string();
        let canonical = {
            let cat = self.inner.catalog.lock().await;
            cat.resolve(tag).map(|(_, c)| c).unwrap_or_else(|| tag.to_string())
        };
        let mut pools = self.inner.pools.lock().await;
        pools.remove(&canonical);
        if canonical != tag {
            pools.remove(tag);
        }
        let with_latest = format!("{stem}:latest");
        if with_latest != canonical && with_latest != tag {
            pools.remove(&with_latest);
        }
        if stem != canonical && stem != tag {
            pools.remove(&stem);
        }
        drop(pools);
        self.emit(EngineEvent::ModelUnloaded { tag: canonical });
        Ok(())
    }
}

// ─────────────────────────── Replica pool (M5b) ────────────────────────────────
//
// candle's quantized forward is `&mut self` with an internal, per-instance KV
// cache, so two generations cannot share one model instance — each concurrent
// request needs its own replica. The pool grows lazily (empty until demand
// arrives), caps at `max`, and FIFO-queues beyond that. This lifts the pre-M5b
// per-tag serialization: with `max > 1`, several requests generate in parallel
// (memory permitting); at the cap they queue fairly instead of head-of-line
// blocking. True token-level batching (shared matmul across sequences) remains
// blocked on candle's single-sequence quantized KV — see ACRoad.md §7 (M5b).
//
// The pool is generic over the replica handle `T` so its bookkeeping can be
// unit-tested with a trivial stand-in instead of a full LoadedEntry. The server
// instantiates it at `T = Replica` (see the alias below).

/// One built-model replica handle: an `Arc` around a `std::sync::Mutex<LoadedEntry>`
/// (generation is synchronous `&mut self`).
pub type Replica = Arc<std::sync::Mutex<LoadedEntry>>;

/// What [`ReplicaPool::acquire`] resolved to.
pub enum AcquireOutcome<T> {
    /// A free replica is ready to use right now.
    Ready(T),
    /// Nothing free, but the pool may grow: the caller builds a replica and
    /// reports it back via [`ReplicaPool::adopt`].
    Build,
    /// At capacity with none free — await the receiver for a released replica.
    Wait(tokio::sync::oneshot::Receiver<T>),
}

/// A pool of up to `max` replicas for one tag, generic over the handle type `T`.
///
/// Uses a `std::sync::Mutex` for its inner state, held only for the brief
/// synchronous bookkeeping (never across an await) so the `release` path is
/// callable directly from the blocking generation thread.
pub struct ReplicaPool<T: Clone + Send + 'static> {
    max: usize,
    inner: std::sync::Mutex<PoolInner<T>>,
}

struct PoolInner<T> {
    /// Every replica ever built for this tag.
    replicas: Vec<T>,
    /// Replicas currently free (a subset of `replicas`).
    free: std::collections::VecDeque<T>,
    /// Build slots reserved but not yet adopted (caps concurrent builds).
    in_flight_builds: usize,
    /// FIFO requests waiting for a replica, woken on release.
    waiters: std::collections::VecDeque<tokio::sync::oneshot::Sender<T>>,
}

impl<T: Clone + Send + 'static> ReplicaPool<T> {
    pub fn new(max: usize) -> Self {
        Self {
            max: max.max(1),
            inner: std::sync::Mutex::new(PoolInner {
                replicas: Vec::new(),
                free: std::collections::VecDeque::new(),
                in_flight_builds: 0,
                waiters: std::collections::VecDeque::new(),
            }),
        }
    }

    /// Max replicas this pool will hold.
    pub fn max(&self) -> usize {
        self.max
    }

    /// Number of replicas built so far (for tests / introspection).
    pub fn built_count(&self) -> usize {
        self.inner.lock().unwrap().replicas.len()
    }

    /// Number of requests currently queued (for tests / introspection).
    pub fn waiter_count(&self) -> usize {
        self.inner.lock().unwrap().waiters.len()
    }

    /// Try to obtain a replica. Never blocks on a generation — only briefly on
    /// the internal mutex. The caller awaits the [`AcquireOutcome::Wait`]
    /// receiver separately if returned.
    pub fn acquire(&self) -> AcquireOutcome<T> {
        let mut inner = self.inner.lock().unwrap();
        if let Some(r) = inner.free.pop_front() {
            return AcquireOutcome::Ready(r);
        }
        // Nothing free. Capacity consumed = built replicas (all checked out,
        // since `free` is empty) + reserved build slots.
        let consumed = inner.replicas.len() + inner.in_flight_builds;
        if consumed < self.max {
            inner.in_flight_builds += 1;
            AcquireOutcome::Build
        } else {
            let (tx, rx) = tokio::sync::oneshot::channel();
            inner.waiters.push_back(tx);
            AcquireOutcome::Wait(rx)
        }
    }

    /// Register a freshly-built replica and take it for use (after a successful
    /// build following [`AcquireOutcome::Build`]).
    pub fn adopt(&self, replica: T) -> T {
        let mut inner = self.inner.lock().unwrap();
        inner.in_flight_builds = inner.in_flight_builds.saturating_sub(1);
        inner.replicas.push(replica.clone());
        // Handed straight to the caller — not added to `free`.
        replica
    }

    /// A reserved build slot came back empty (build failed) — release it.
    pub fn build_failed(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.in_flight_builds = inner.in_flight_builds.saturating_sub(1);
    }

    /// Return a replica to the pool, handing it to the next waiter if any. Sync
    /// — safe to call from the blocking generation thread when a run finishes.
    pub fn release(&self, replica: T) {
        let mut inner = self.inner.lock().unwrap();
        // Hand to the oldest still-live waiter. We send a clone so a dropped
        // waiter (Err — client went away) doesn't lose the replica: we keep the
        // original and try the next waiter / fall back to `free`.
        while let Some(tx) = inner.waiters.pop_front() {
            if tx.send(replica.clone()).is_ok() {
                return;
            }
        }
        inner.free.push_back(replica);
    }
}

/// RAII handle to a checked-out replica. Dropping it returns the replica to the
/// pool (or hands it to the next queued request).
pub struct ReplicaHandle<T: Clone + Send + 'static> {
    pool: Arc<ReplicaPool<T>>,
    replica: Option<T>,
}

impl<T: Clone + Send + 'static> ReplicaHandle<T> {
    /// The acquired replica, held for the duration of one generation.
    pub fn replica(&self) -> &T {
        self.replica
            .as_ref()
            .expect("replica handle used after release")
    }
}

impl<T: Clone + Send + 'static> Drop for ReplicaHandle<T> {
    fn drop(&mut self) {
        if let Some(r) = self.replica.take() {
            self.pool.release(r);
        }
    }
}

/// Build the axum router for the Phoenix-compatible API.
pub fn app(state: ServerState) -> Router {
    Router::new()
        .route("/api/tags", get(tags::list))
        .route("/api/chat", post(chat::chat))
        .route("/api/stats", get(stats::stats))
        .route("/api/telemetry/status", get(telemetry_status))
        .with_state(state)
}

/// `GET /api/telemetry/status` — reports whether the Prometheus collector push
/// is configured and the cached hardware snapshot. Useful for sanity-checking a
/// tester's setup without running a full generation.
async fn telemetry_status(
    axum::extract::State(state): axum::extract::State<ServerState>,
) -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({
        "collector_configured": telemetry::collector_url().is_some(),
        "backend": state.backend_name(),
        "hardware": *state.hardware(),
    }))
}

/// Run the HTTP server on the given port (default [`DEFAULT_PORT`][crate::DEFAULT_PORT]).
pub async fn serve(port: u16, state: ServerState) -> Result<()> {
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| Error::Server(format!("bind {addr}: {e}")))?;
    tracing::info!(%addr, "AmberCore server listening");
    axum::serve(listener, app(state))
        .await
        .map_err(|e| Error::Server(format!("serve: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The pool is generic over its handle type `T`, so we can exercise all the
    // bookkeeping (build/free/wait/FIFO/capacity) with a trivial stand-in
    // instead of a full LoadedEntry. Each "replica" is a distinct Arc<u32>.
    type TestReplica = Arc<u32>;

    fn r(n: u32) -> TestReplica {
        Arc::new(n)
    }

    #[tokio::test]
    async fn engine_events_reach_subscribers() {
        // The channel is the host's window onto SILENT recoveries (OOM CPU
        // fallback above all) — verify the fan-out works with a minimal state.
        let dir = std::env::temp_dir().join(format!("pa-engine-ev-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let state = ServerState::new(
            Catalog::load(&dir).expect("catalog on an empty temp dir"),
            Box::new(crate::backend::CpuBackend::default()),
            1,
        );
        let mut rx = state.subscribe();
        state.emit(EngineEvent::OomFallback { tag: "rwkv7".into() });
        state.emit(EngineEvent::ModelLoaded {
            tag: "rwkv7".into(),
            device: "cpu".into(),
            dur_ms: 12,
        });
        assert!(matches!(
            rx.try_recv(),
            Ok(EngineEvent::OomFallback { ref tag }) if tag == "rwkv7"
        ));
        assert!(matches!(
            rx.try_recv(),
            Ok(EngineEvent::ModelLoaded { device: ref d, dur_ms: 12, .. }) if d == "cpu"
        ));
        // No subscriber = sends stay free no-ops (this must not panic).
        let _ = state.subscribe();
        state.emit(EngineEvent::ModelUnloaded { tag: "rwkv7".into() });
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pool_max_clamped_to_at_least_one() {
        let p = ReplicaPool::<TestReplica>::new(0);
        assert_eq!(p.max(), 1);
    }

    #[test]
    fn first_acquire_requests_a_build() {
        let p = ReplicaPool::<TestReplica>::new(2);
        // Empty pool → nothing free, room to grow → Build. Nothing built yet.
        assert!(matches!(p.acquire(), AcquireOutcome::Build));
        assert_eq!(p.built_count(), 0);
        let a = p.adopt(r(1));
        assert_eq!(p.built_count(), 1);
        assert_eq!(*a, 1);
    }

    #[test]
    fn release_returns_replica_to_free_for_reuse() {
        let p = ReplicaPool::<TestReplica>::new(2);
        let a = p.adopt(r(7));
        p.release(a);
        // Acquire must hand back the freed replica (not request a new build).
        match p.acquire() {
            AcquireOutcome::Ready(got) => assert_eq!(*got, 7),
            _ => panic!("expected Ready after a release"),
        }
        assert_eq!(p.built_count(), 1); // still only one built — reused, not grown
    }

    #[test]
    fn growth_caps_at_max_then_queues() {
        let p = ReplicaPool::<TestReplica>::new(2);
        let _a = p.adopt(r(1));
        let _b = p.adopt(r(2));
        // Both replicas built & held; pool at capacity, none free → Wait.
        let waiters_before = p.waiter_count();
        assert!(matches!(p.acquire(), AcquireOutcome::Wait(_)));
        assert_eq!(p.waiter_count(), waiters_before + 1);
    }

    #[test]
    fn build_failed_frees_a_growth_slot() {
        let p = ReplicaPool::<TestReplica>::new(1);
        assert!(matches!(p.acquire(), AcquireOutcome::Build)); // reserves the slot
        // While the build is in flight, a second request must queue (at capacity).
        assert!(matches!(p.acquire(), AcquireOutcome::Wait(_)));
        p.build_failed(); // slot returned
        // Now the pool can build again.
        assert!(matches!(p.acquire(), AcquireOutcome::Build));
    }

    #[tokio::test]
    async fn fifo_waiters_served_in_order() {
        // max=2: build two replicas and hold both, then queue two waiters.
        let p = Arc::new(ReplicaPool::<TestReplica>::new(2));
        let a = p.adopt(r(1));
        let b = p.adopt(r(2));
        // Both held (we own a, b). Queue two waiters in order.
        let rx1 = match p.acquire() {
            AcquireOutcome::Wait(rx) => rx,
            _ => panic!("expected Wait"),
        };
        let rx2 = match p.acquire() {
            AcquireOutcome::Wait(rx) => rx,
            _ => panic!("expected Wait"),
        };
        // Release in reverse order; the FIRST waiter must get the FIRST-released
        // replica (FIFO), the second waiter the second.
        p.release(b);
        p.release(a);
        let got1 = rx1.await.expect("waiter 1 woken");
        let got2 = rx2.await.expect("waiter 2 woken");
        assert_eq!(*got1, 2); // first waiter ← first released (b)
        assert_eq!(*got2, 1); // second waiter ← second released (a)
    }

    /// `remove_model` drops every spelling of a tag (manifest bare stem +
    /// scan-derived `:latest`), persists the removal to `manifest.json`, and
    /// tolerates being handed either spelling.
    #[tokio::test]
    async fn remove_model_drops_all_spellings_and_persists() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/mymodel-Q4.gguf"), b"x").unwrap();
        let cat = Catalog::load(dir.path()).unwrap();
        let backend =
            crate::backend::resolve_backend(crate::backend::DeviceChoice::Cpu).unwrap();
        let state = ServerState::new(cat, backend, 1);
        // Manifest-registered under the bare stem — Phoenix's pull spelling —
        // while the scan also derived `mymodel-Q4:latest` from the file.
        state
            .register_entry(CatalogEntry {
                tag: "mymodel-Q4".into(),
                file: "sub/mymodel-Q4.gguf".into(),
                arch: None,
            })
            .await
            .unwrap();
        let tags = state.tags().await;
        assert!(tags.contains(&"mymodel-Q4".to_string()));
        assert!(tags.contains(&"mymodel-Q4:latest".to_string()));

        // Remove via the :latest spelling → both spellings go.
        state.remove_model("mymodel-Q4:latest").await.unwrap();
        assert!(state.tags().await.is_empty());

        // The manifest entry was persisted away.
        let raw = std::fs::read_to_string(dir.path().join("manifest.json")).unwrap();
        assert!(!raw.contains("mymodel-Q4"), "manifest should not mention the model: {raw}");

        // Removing an unknown tag is a no-op success, not an error.
        state.remove_model("never-existed").await.unwrap();
    }
}
