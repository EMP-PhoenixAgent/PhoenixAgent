//! Dispatching provider — routes trait calls to the active backend.
//!
//! This is the single `Arc<dyn ModelProvider>` that [`crate::web::state::WebState`]
//! owns and clones into the agent runtime + health monitor. Because routing lives
//! behind an `RwLock` *inside* this shared Arc, switching the active backend
//! (local Ollama / local AmberCore / a cloud provider) reaches the runtime and
//! health monitor on their next call — no re-spawn, no restart.
//!
//! Why this exists: the previous design relied on a single [`OllamaProvider`]
//! whose `base_url` was an `RwLock`, which worked only because Ollama and
//! AmberCore share one wire protocol. Cloud providers use a *different* protocol
//! (OpenAI Chat Completions + Bearer auth + SSE), so we need a runtime route
//! that picks the concrete backend per call.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::{mpsc, RwLock};

use crate::error::{PhoenixError, Result};
use crate::model::{
    ambercore_embedded::EmbeddedAmberCore, openai::OpenAiProvider, ollama::OllamaProvider,
    ChatEvent, ChatRequest, ModelProvider, ProviderStats, Usage,
};

/// Which local Ollama-compatible server the local route points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalBackend {
    Ollama,
    AmberCore,
}

impl LocalBackend {
    pub fn as_str(&self) -> &'static str {
        match self {
            LocalBackend::Ollama => "ollama",
            LocalBackend::AmberCore => "ambercore",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "ollama" => Some(Self::Ollama),
            "ambercore" => Some(Self::AmberCore),
            _ => None,
        }
    }
}

/// The active cloud endpoint (a registered provider).
#[derive(Debug, Clone)]
pub struct CloudRoute {
    pub provider_id: i64,
    pub base_url: String,
    pub api_key: String,
}

/// The currently active route. Mutated by the command layer when the user clicks
/// a "Run" button in the Models panel.
#[derive(Debug, Clone)]
pub enum ActiveRoute {
    /// A local Ollama-compatible server (Ollama on `:11434` or AmberCore on
    /// `:42069`). The URL is owned by the inner [`OllamaProvider`].
    Local { backend: LocalBackend },
    /// A cloud OpenAI-compatible provider. Credentials are pushed into the inner
    /// [`OpenAiProvider`] whenever the route changes.
    Cloud { route: CloudRoute },
}

impl Default for ActiveRoute {
    fn default() -> Self {
        ActiveRoute::Local {
            backend: LocalBackend::Ollama,
        }
    }
}

/// Live runtime metrics for the active route, measured at the dispatch layer so
/// every backend (embedded AmberCore, remote AmberCore, Ollama, cloud APIs)
/// reports the same numbers. TTFT = chat start → first generation event; TBT =
/// average gap between generation events; T/s = output tokens ÷ stream duration.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct RuntimeMetrics {
    /// True while at least one generation is in flight (main agent or sub-agent).
    pub busy: bool,
    /// Number of generation streams currently in flight.
    #[serde(skip)]
    pub in_flight: u32,
    /// Last completed turn's throughput in tokens/second.
    pub tokens_per_sec: Option<f64>,
    /// Last completed turn's time-to-first-token, in milliseconds.
    pub ttft_ms: Option<f64>,
    /// Last completed turn's average time-between-tokens, in milliseconds.
    pub tbt_avg_ms: Option<f64>,
    /// Token counts from the last completed turn, when the backend reported them.
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
}

/// Routes [`ModelProvider`] calls to whichever backend is currently active.
///
/// Holds one [`OllamaProvider`] (HTTP — Ollama, or a *remote* AmberCore server),
/// one [`OpenAiProvider`] (reused for all cloud providers, reconfigured per
/// provider), and one [`EmbeddedAmberCore`] — the in-process engine compiled
/// into the binary, which is what "local AmberCore" means now. The active route
/// is an `RwLock`; `Local { backend: AmberCore }` resolves to the embedded
/// engine or the HTTP provider depending on [`Self::set_local_embedded`] vs
/// [`Self::set_local`].
pub struct DispatchProvider {
    route: RwLock<ActiveRoute>,
    /// True while "local AmberCore" means the in-process engine (the default);
    /// false while it points at a remote server over HTTP.
    embedded_ambercore: RwLock<bool>,
    local: OllamaProvider,
    cloud: OpenAiProvider,
    embedded: EmbeddedAmberCore,
    /// Dispatch-layer timing metrics shared by every instrumented chat stream.
    metrics: Arc<Mutex<RuntimeMetrics>>,
}

impl DispatchProvider {
    /// Build with a starting local URL (the active backend's URL), route, and
    /// the embedded AmberCore engine (constructed once at app launch).
    pub fn new(
        local_url: impl Into<String>,
        route: ActiveRoute,
        embedded: EmbeddedAmberCore,
    ) -> Self {
        let local = OllamaProvider::new(local_url);
        let cloud = OpenAiProvider::new();
        Self {
            route: RwLock::new(route),
            embedded_ambercore: RwLock::new(true),
            local,
            cloud,
            embedded,
            metrics: Arc::new(Mutex::new(RuntimeMetrics::default())),
        }
    }

    /// Snapshot of the dispatch-layer runtime metrics (for the health bar UI).
    pub fn runtime_metrics(&self) -> RuntimeMetrics {
        self.metrics.lock().expect("runtime metrics lock").clone()
    }

    /// Merged stats view for the UI: engine-reported throughput (decode-only,
    /// the honest generation speed) wins when the backend exposes one; timing
    /// metrics + busy come from the dispatch-layer instrumentation.
    pub async fn merged_stats(&self) -> ProviderStats {
        let embedded = match self.route.read().await.clone() {
            ActiveRoute::Local { backend: LocalBackend::AmberCore } => {
                *self.embedded_ambercore.read().await
            }
            _ => false,
        };
        let engine = if embedded {
            self.embedded.stats().await
        } else {
            match self.route.read().await.clone() {
                // Only AmberCore serves /api/stats; Ollama degrades to None inside.
                ActiveRoute::Local { .. } => self.local.stats().await,
                ActiveRoute::Cloud { .. } => None,
            }
        };
        let m = self.runtime_metrics();
        ProviderStats {
            tokens_per_sec: engine
                .as_ref()
                .and_then(|e| e.tokens_per_sec)
                .or(m.tokens_per_sec),
            ttft_ms: m.ttft_ms,
            tbt_avg_ms: m.tbt_avg_ms,
            busy: m.busy,
        }
    }

    /// Read the current route (cloned).
    pub async fn route(&self) -> ActiveRoute {
        self.route.read().await.clone()
    }

    /// Switch to a local HTTP backend (Ollama, or a *remote* AmberCore server).
    /// Flips the inner Ollama provider's URL and sets the route. `local_url` is
    /// the resolved URL for `backend`.
    pub async fn set_local(&self, backend: LocalBackend, local_url: String) {
        self.local.set_endpoint(local_url).await;
        *self.embedded_ambercore.write().await = false;
        *self.route.write().await = ActiveRoute::Local { backend };
    }

    /// Switch "local AmberCore" to the **in-process engine** — no HTTP hop, no
    /// `ambercore serve` process. This is the default local mode.
    pub async fn set_local_embedded(&self) {
        *self.embedded_ambercore.write().await = true;
        *self.route.write().await = ActiveRoute::Local {
            backend: LocalBackend::AmberCore,
        };
    }

    /// Preload a model into the embedded engine's replica pool (background).
    /// Called at launch with the last-used model so the first message is fast.
    pub async fn warm_ambercore_model(&self, model: &str) {
        self.embedded.warm_model(model).await;
    }

    /// The embedded engine (for catalog reloads after pulls/registers).
    pub fn embedded(&self) -> &EmbeddedAmberCore {
        &self.embedded
    }

    /// Switch to a cloud provider. Pushes the endpoint + key into the inner
    /// OpenAI provider and sets the route.
    pub async fn set_cloud(&self, route: CloudRoute) {
        self.cloud
            .configure(route.base_url.clone(), route.api_key.clone())
            .await;
        *self.route.write().await = ActiveRoute::Cloud { route };
    }

    /// Swap the local URL without changing which local backend is active.
    /// (Used by the legacy `set_backend` command and the health probe.)
    pub async fn set_local_url(&self, url: String) {
        self.local.set_endpoint(url).await;
    }
}

#[async_trait]
impl ModelProvider for DispatchProvider {
    async fn list_models(&self) -> Result<Vec<String>> {
        let embedded = match self.route.read().await.clone() {
            ActiveRoute::Local { backend: LocalBackend::AmberCore } => {
                *self.embedded_ambercore.read().await
            }
            _ => false,
        };
        if embedded {
            self.embedded.list_models().await
        } else {
            match self.route.read().await.clone() {
                ActiveRoute::Local { .. } => self.local.list_models().await,
                ActiveRoute::Cloud { .. } => self.cloud.list_models().await,
            }
        }
    }

    async fn chat(
        &self,
        model: &str,
        request: ChatRequest,
    ) -> Result<mpsc::Receiver<ChatEvent>> {
        let embedded = match self.route.read().await.clone() {
            ActiveRoute::Local { backend: LocalBackend::AmberCore } => {
                *self.embedded_ambercore.read().await
            }
            _ => false,
        };
        let inner = if embedded {
            self.embedded.chat(model, request).await?
        } else {
            match self.route.read().await.clone() {
                ActiveRoute::Local { .. } => self.local.chat(model, request).await?,
                ActiveRoute::Cloud { .. } => self.cloud.chat(model, request).await?,
            }
        };
        Ok(instrument_stream(inner, self.metrics.clone()))
    }

    /// Swap the local endpoint at runtime. Kept for compatibility with code that
    /// treats the provider as opaque; cloud switches go through `set_cloud`.
    async fn set_endpoint(&self, url: String) {
        self.set_local_url(url).await;
    }

    async fn stats(&self) -> Option<ProviderStats> {
        Some(self.merged_stats().await)
    }
}

/// Convenience for tests and the command layer: a dispatch provider is always
/// shared behind an `Arc<dyn ModelProvider>`. Uses the default models dir for
/// the embedded engine (no model needs to exist for construction).
pub fn shared(
    local_url: impl Into<String>,
    route: ActiveRoute,
) -> Arc<dyn ModelProvider> {
    let embedded = EmbeddedAmberCore::new(None)
        .expect("embedded AmberCore engine (default models dir) must construct");
    Arc::new(DispatchProvider::new(local_url, route, embedded)) as Arc<dyn ModelProvider>
}

/// Error type for invalid route arguments surfaced from commands.
#[allow(dead_code)]
pub fn bad_route(msg: impl Into<String>) -> PhoenixError {
    PhoenixError::Model(msg.into())
}

/// Wrap a provider's event stream with wall-clock instrumentation and forward
/// every event untouched. This is what makes runtime metrics work identically
/// for every backend (embedded AmberCore, remote AmberCore, Ollama, cloud):
/// TTFT = chat start → first generation event, TBT = average gap between
/// generation events, T/s = output tokens ÷ stream duration.
fn instrument_stream(
    mut rx: mpsc::Receiver<ChatEvent>,
    metrics: Arc<Mutex<RuntimeMetrics>>,
) -> mpsc::Receiver<ChatEvent> {
    let (tx, out) = mpsc::channel(32);
    // Mark busy synchronously so the flag is true the moment chat() returns.
    {
        let mut m = metrics.lock().expect("runtime metrics lock");
        m.in_flight += 1;
        m.busy = true;
    }
    tokio::spawn(async move {
        let start = std::time::Instant::now();
        let mut first_event: Option<std::time::Instant> = None;
        let mut last_event: Option<std::time::Instant> = None;
        let mut intervals_ms: Vec<f64> = Vec::new();
        let mut usage: Option<Usage> = None;
        while let Some(ev) = rx.recv().await {
            match &ev {
                ChatEvent::Delta(_) | ChatEvent::Reasoning(_) => {
                    let now = std::time::Instant::now();
                    if first_event.is_none() {
                        first_event = Some(now);
                    } else if let Some(last) = last_event {
                        intervals_ms.push(now.duration_since(last).as_secs_f64() * 1000.0);
                    }
                    last_event = Some(now);
                }
                ChatEvent::Done(u) => usage = Some(u.clone()),
                _ => {}
            }
            if tx.send(ev).await.is_err() {
                break; // consumer gone; still record metrics below
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let ttft_ms = first_event.map(|t| t.duration_since(start).as_secs_f64() * 1000.0);
        let tbt_avg_ms = if intervals_ms.is_empty() {
            None
        } else {
            Some(intervals_ms.iter().sum::<f64>() / intervals_ms.len() as f64)
        };
        let tokens_per_sec = usage
            .as_ref()
            .filter(|u| u.output_tokens > 0)
            .filter(|_| elapsed > 0.0)
            .map(|u| u.output_tokens as f64 / elapsed);
        let mut m = metrics.lock().expect("runtime metrics lock");
        m.in_flight = m.in_flight.saturating_sub(1);
        m.busy = m.in_flight > 0;
        // Only overwrite what this stream actually measured, so an aborted
        // stream (error mid-turn) keeps the previous good numbers.
        if let Some(v) = ttft_ms {
            m.ttft_ms = Some(v);
        }
        if let Some(v) = tbt_avg_ms {
            m.tbt_avg_ms = Some(v);
        }
        if let Some(v) = tokens_per_sec {
            m.tokens_per_sec = Some(v);
        }
        if let Some(u) = usage {
            m.input_tokens = Some(u.input_tokens);
            m.output_tokens = Some(u.output_tokens);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn instrument_stream_records_ttft_tbt_and_busy() {
        let metrics = Arc::new(Mutex::new(RuntimeMetrics::default()));
        let (tx, rx) = mpsc::channel(8);
        let mut out = instrument_stream(rx, metrics.clone());
        assert!(metrics.lock().unwrap().busy, "busy flips on at stream start");

        tx.send(ChatEvent::Reasoning("thinking".into()))
            .await
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        tx.send(ChatEvent::Delta("a".into())).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        tx.send(ChatEvent::Delta("b".into())).await.unwrap();
        tx.send(ChatEvent::Done(Usage { input_tokens: 3, output_tokens: 2 }))
            .await
            .unwrap();
        drop(tx);

        while out.recv().await.is_some() {}
        let m = metrics.lock().unwrap().clone();
        assert!(!m.busy, "busy flips off when the stream ends");
        assert!(m.ttft_ms.expect("ttft recorded") >= 0.0);
        // One timed gap between the two post-first events.
        assert!(m.tbt_avg_ms.expect("tbt recorded") > 0.0);
        assert_eq!(m.output_tokens, Some(2));
        assert_eq!(m.input_tokens, Some(3));
        assert!(m.tokens_per_sec.expect("tok/s computed") > 0.0);
    }

    #[tokio::test]
    async fn instrument_stream_forwards_events_and_clears_busy_on_error() {
        let metrics = Arc::new(Mutex::new(RuntimeMetrics::default()));
        let (tx, rx) = mpsc::channel(8);
        let mut out = instrument_stream(rx, metrics.clone());
        tx.send(ChatEvent::Delta("x".into())).await.unwrap();
        tx.send(ChatEvent::Error("boom".into())).await.unwrap();
        drop(tx);

        assert!(matches!(out.recv().await, Some(ChatEvent::Delta(d)) if d == "x"));
        assert!(matches!(out.recv().await, Some(ChatEvent::Error(_))));
        assert!(out.recv().await.is_none());
        assert!(!metrics.lock().unwrap().busy, "busy clears after an errored stream");
    }

    #[tokio::test]
    async fn concurrent_streams_keep_busy_until_both_finish() {
        let metrics = Arc::new(Mutex::new(RuntimeMetrics::default()));
        let (tx1, rx1) = mpsc::channel(4);
        let (tx2, rx2) = mpsc::channel(4);
        let mut out1 = instrument_stream(rx1, metrics.clone());
        let mut out2 = instrument_stream(rx2, metrics.clone());
        tx1.send(ChatEvent::Delta("1".into())).await.unwrap();
        tx2.send(ChatEvent::Delta("2".into())).await.unwrap();
        drop(tx1);
        while out1.recv().await.is_some() {}
        assert!(metrics.lock().unwrap().busy, "still busy while stream 2 runs");
        drop(tx2);
        while out2.recv().await.is_some() {}
        assert!(!metrics.lock().unwrap().busy, "busy clears when the last stream ends");
    }
}
