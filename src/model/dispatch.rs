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

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{mpsc, RwLock};

use crate::error::{PhoenixError, Result};
use crate::model::{
    ambercore_embedded::EmbeddedAmberCore, openai::OpenAiProvider, ollama::OllamaProvider,
    ChatEvent, ChatRequest, ModelProvider, ProviderStats,
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
        if embedded {
            self.embedded.chat(model, request).await
        } else {
            match self.route.read().await.clone() {
                ActiveRoute::Local { .. } => self.local.chat(model, request).await,
                ActiveRoute::Cloud { .. } => self.cloud.chat(model, request).await,
            }
        }
    }

    /// Swap the local endpoint at runtime. Kept for compatibility with code that
    /// treats the provider as opaque; cloud switches go through `set_cloud`.
    async fn set_endpoint(&self, url: String) {
        self.set_local_url(url).await;
    }

    async fn stats(&self) -> Option<ProviderStats> {
        let embedded = match self.route.read().await.clone() {
            ActiveRoute::Local { backend: LocalBackend::AmberCore } => {
                *self.embedded_ambercore.read().await
            }
            _ => false,
        };
        if embedded {
            self.embedded.stats().await
        } else {
            match self.route.read().await.clone() {
                // Only AmberCore serves /api/stats; Ollama degrades to None inside.
                ActiveRoute::Local { .. } => self.local.stats().await,
                ActiveRoute::Cloud { .. } => None,
            }
        }
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
