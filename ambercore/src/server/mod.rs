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

/// End-of-turn token strings per architecture family. Tokens absent from the
/// loaded tokenizer are skipped, and the GGUF's `<arch>.eos_token_id` is added
/// separately (by id, so models with unusual EOS strings still stop).
fn arch_stop_markers(arch: &str) -> &'static [&'static str] {
    match arch {
        "gemma" | "gemma2" | "gemma3" => &["<end_of_turn>", "<|endoftext|>"],
        "phi2" => &["<|endoftext|>"],
        "phi3" => &["<|end|>", "<|endoftext|>"],
        "glm4" => &["<|user|>", "<|observation|>", "<|endoftext|>"],
        "mixtral" => &["</s>", "<|end_of_text|>"],
        "llama" => &["<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>"],
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
            }),
        }
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

    /// Resolve a tag to its GGUF and build a fresh [`LoadedEntry`] (load GGUF,
    /// build the model, load the tokenizer, resolve stop tokens). The candle
    /// parsing runs on a blocking thread. Factored out so the first replica and
    /// later pool-grown replicas share one build path.
    async fn build_loaded_entry(&self, tag: &str) -> Result<LoadedEntry> {
        let (gguf_path, _arch_hint) = {
            let cat = self.inner.catalog.lock().await;
            let entry = cat
                .get(tag)
                .ok_or_else(|| Error::NotFound(format!("model tag '{tag}' not in catalog")))?;
            (cat.resolve_path(entry), entry.arch.clone())
        };
        tracing::info!(tag, "building model replica from {}", gguf_path.display());
        let device = self.inner.backend.device()?;
        let backend_name = self.inner.backend.name().to_string();
        let tag_owned = tag.to_string();
        tokio::task::spawn_blocking(move || -> Result<LoadedEntry> {
            let mut loaded = LoadedModel::load(&gguf_path)?;
            let arch = loaded.arch.clone();
            // Surface the trained context length from the GGUF metadata
            // (`<arch>.context_length`). Falls back to a large default if absent.
            let context_length = loaded
                .meta_str(&format!("{arch}.context_length"))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(4096);
            let tokenizer = TokenizerWrapper::load_next_to(&gguf_path)?;
            let eos_meta = loaded
                .meta_str(&format!("{arch}.eos_token_id"))
                .and_then(|s| s.parse::<u32>().ok());
            let model = build_model(&mut loaded, &device)?;
            // Stop tokens: the GGUF's own `<arch>.eos_token_id` plus the
            // per-architecture end-of-turn markers (absent tokens are skipped,
            // so a family's list can include markers only some models carry).
            let mut stop_tokens = Vec::new();
            if let Some(id) = eos_meta {
                stop_tokens.push(id);
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
        })
        .await
        .map_err(|e| Error::Server(format!("load task join: {e}")))?
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
        let pool = self.pool_for(tag).await?;
        loop {
            match pool.acquire() {
                AcquireOutcome::Ready(r) => {
                    return Ok(ReplicaHandle { pool: pool.clone(), replica: Some(r) });
                }
                AcquireOutcome::Build => {
                    // Grow the pool. Build OUTSIDE the pool lock (expensive).
                    let entry = match self.build_loaded_entry(tag).await {
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
}
