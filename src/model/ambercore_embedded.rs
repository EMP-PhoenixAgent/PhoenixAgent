//! Embedded (in-process) AmberCore provider.
//!
//! AmberCore — the pure-Rust LLM runner — is compiled *into* the Phoenix binary
//! (`ambercore = { path = "ambercore" }`) and driven directly through its
//! [`generate_events`] API. There is no HTTP hop and no separate `ambercore
//! serve` process: launching Phoenix Agent launches the model engine.
//!
//! - [`EmbeddedAmberCore::new`] builds the engine once (catalog + replica pool +
//!   compute backend, resolved `auto`: CUDA → Metal → CPU depending on the
//!   build's enabled features).
//! - [`ModelProvider::chat`] maps Phoenix's [`ChatRequest`] onto AmberCore's
//!   protocol types and forwards the resulting [`GenEvent`] stream as
//!   [`ChatEvent`]s — reasoning (`<think>`) included, so the UI's thinking block
//!   works exactly as it does over HTTP.
//! - Model replicas load lazily on first use; [`EmbeddedAmberCore::warm_model`]
//!   preloads one in the background so the first message is answered fast.
//!
//! GPU acceleration is opt-in at build time (`--features ambercore-cuda` /
//! `ambercore-metal`); the shipped installers are portable CPU builds.

use std::path::PathBuf;

use async_trait::async_trait;
use tokio::sync::mpsc;

use ambercore::backend::{resolve_backend, DeviceChoice};
use ambercore::catalog::{default_models_dir, Catalog};
use ambercore::server::chat::{generate_events, GenEvent};
use ambercore::server::protocol as ac;
use ambercore::server::ServerState;

use crate::error::{PhoenixError, Result};
use crate::model::{
    ChatEvent, ChatRequest, ModelProvider, ProviderStats, ToolCall, Usage,
};

/// The in-process AmberCore engine. Cheap to share (`ServerState` is an `Arc`).
pub struct EmbeddedAmberCore {
    state: ServerState,
}

impl EmbeddedAmberCore {
    /// Build the engine against a models directory (`None` = AmberCore's native
    /// `~/.ambercore/models`). The directory is created if missing so pulls and
    /// `ambercore register` have a home; an unreadable directory is a hard error.
    pub fn new(models_dir: Option<PathBuf>) -> Result<Self> {
        let dir = match models_dir {
            Some(d) => d,
            None => default_models_dir()
                .map_err(|e| PhoenixError::Model(format!("ambercore models dir: {e}")))?,
        };
        std::fs::create_dir_all(&dir)
            .map_err(|e| PhoenixError::Model(format!("create {}: {e}", dir.display())))?;
        let catalog = Catalog::load(&dir)
            .map_err(|e| PhoenixError::Model(format!("ambercore catalog: {e}")))?;
        // `auto` resolves to CUDA/Metal when compiled in AND available, else CPU
        // — it never fails, so a GPU-less machine just runs on CPU.
        let backend = resolve_backend(DeviceChoice::Auto)
            .map_err(|e| PhoenixError::Model(format!("ambercore backend: {e}")))?;
        tracing::info!(
            backend = backend.name(),
            models = catalog.tags().len(),
            "embedded AmberCore engine ready (models load lazily)"
        );
        Ok(Self {
            state: ServerState::new(catalog, backend, 1),
        })
    }

    /// Access the shared engine state (used for warm-ups / catalog reloads).
    pub fn state(&self) -> &ServerState {
        &self.state
    }

    /// Preload a model replica in the background so the first chat using it
    /// doesn't pay the GGUF load. The built replica stays pooled for reuse.
    /// Failures (e.g. the tag isn't in the catalog yet) are logged, not raised.
    pub async fn warm_model(&self, model: &str) {
        match self.state.acquire_replica(model).await {
            Ok(_handle) => tracing::info!(model, "embedded AmberCore: model warmed (pooled)"),
            Err(e) => tracing::warn!(model, "embedded AmberCore: warm-up skipped: {e}"),
        }
        // The handle drops here → the replica returns to the pool's free list.
    }

    /// Reload the catalog after models were pulled/registered.
    pub async fn reload_catalog(&self, models_dir: PathBuf) -> Result<()> {
        self.state
            .reload_catalog(models_dir)
            .await
            .map_err(|e| PhoenixError::Model(format!("ambercore catalog reload: {e}")))
    }

    /// Register a downloaded GGUF under a tag, in-process (persists to the
    /// models dir's `manifest.json`; no `ambercore register` subprocess).
    /// `file` is the path relative to the models dir.
    pub async fn register_model(&self, tag: &str, file: &str) -> Result<()> {
        let entry = ambercore::catalog::CatalogEntry {
            tag: tag.to_string(),
            file: file.to_string(),
            arch: None,
        };
        self.state
            .register_entry(entry)
            .await
            .map_err(|e| PhoenixError::Model(format!("ambercore register: {e}")))
    }
}

/// Map a Phoenix tool call onto AmberCore's (identical shape, distinct types).
fn to_ac_tool_call(c: &ToolCall) -> ac::ToolCall {
    ac::ToolCall {
        id: c.id.clone(),
        function: ac::FunctionRef {
            name: c.function.name.clone(),
            arguments: c.function.arguments.clone(),
        },
    }
}

/// Map an AmberCore tool call back onto Phoenix's.
fn from_ac_tool_call(c: ac::ToolCall) -> ToolCall {
    ToolCall {
        id: c.id,
        function: crate::model::FunctionRef {
            name: c.function.name,
            arguments: c.function.arguments,
        },
    }
}

#[async_trait]
impl ModelProvider for EmbeddedAmberCore {
    async fn list_models(&self) -> Result<Vec<String>> {
        Ok(self.state.tags().await)
    }

    async fn chat(
        &self,
        model: &str,
        request: ChatRequest,
    ) -> Result<mpsc::Receiver<ChatEvent>> {
        // Map the request onto AmberCore's protocol types.
        let messages = request
            .messages
            .iter()
            .map(|m| ac::ChatMessage {
                role: m.role.as_str().to_string(),
                content: m.content.clone(),
                tool_calls: m
                    .tool_calls
                    .as_ref()
                    .map(|tc| tc.iter().map(to_ac_tool_call).collect()),
                tool: m.tool_name.clone(),
            })
            .collect();
        let tools = request
            .tools
            .iter()
            .map(|t| ac::ToolDef {
                kind: "function".into(),
                function: ac::ToolFunction {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    parameters: t.function.parameters.clone(),
                },
            })
            .collect();
        let ac_req = ac::ChatRequest {
            model: model.to_string(),
            messages,
            tools,
            stream: true,
            temperature: Some(request.temperature),
        };

        // Drive the engine and forward semantic events as ChatEvents.
        let mut ev_rx = generate_events(&self.state, &ac_req)
            .await
            .map_err(|e| PhoenixError::Model(format!("ambercore: {e}")))?;

        let (tx, rx) = mpsc::channel::<ChatEvent>(64);
        tokio::spawn(async move {
            while let Some(ev) = ev_rx.recv().await {
                let chat_ev = match ev {
                    GenEvent::Reasoning(t) => ChatEvent::Reasoning(t),
                    GenEvent::Content(t) => ChatEvent::Delta(t),
                    GenEvent::ToolCalls(tc) => {
                        ChatEvent::ToolCalls(tc.into_iter().map(from_ac_tool_call).collect())
                    }
                    GenEvent::Done { prompt_tokens, output_tokens } => {
                        ChatEvent::Done(Usage {
                            input_tokens: prompt_tokens,
                            output_tokens,
                        })
                    }
                    GenEvent::Error(e) => ChatEvent::Error(e),
                };
                if tx.send(chat_ev).await.is_err() {
                    break; // consumer gone
                }
            }
        });

        Ok(rx)
    }

    /// Live tok/s from the engine's internal counter (powers the health bar).
    async fn stats(&self) -> Option<ProviderStats> {
        self.state
            .last_tokens_per_sec()
            .await
            .map(|tps| ProviderStats {
                tokens_per_sec: Some(tps),
                ..Default::default()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded engine constructs against a temp models dir (empty catalog
    /// is fine) and lists zero models — proving the in-process wiring builds and
    /// answers without any HTTP server or GGUF present.
    #[tokio::test]
    async fn embedded_engine_builds_and_lists() {
        let dir = tempfile::tempdir().unwrap();
        let engine = EmbeddedAmberCore::new(Some(dir.path().to_path_buf())).unwrap();
        let tags = engine.list_models().await.unwrap();
        assert!(tags.is_empty());
    }

    /// Tool calls round-trip through the protocol mapping.
    #[test]
    fn tool_call_mapping_round_trips() {
        let phoenix_call = ToolCall {
            id: Some("call_1".into()),
            function: crate::model::FunctionRef {
                name: "read_file".into(),
                arguments: r#"{"path":"x.rs"}"#.into(),
            },
        };
        let ac_call = to_ac_tool_call(&phoenix_call);
        assert_eq!(ac_call.function.name, "read_file");
        let back = from_ac_tool_call(ac_call);
        assert_eq!(back.function.name, phoenix_call.function.name);
        assert_eq!(back.function.arguments, phoenix_call.function.arguments);
        assert_eq!(back.id.as_deref(), Some("call_1"));
    }
}
