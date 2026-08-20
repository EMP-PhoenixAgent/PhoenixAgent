//! Model abstraction layer.
//!
//! Defines the shared chat types ([`ChatMessage`], [`ChatRole`], [`ToolCall`],
//! [`ToolDef`]) and the [`ModelProvider`] trait that backends implement.
//!
//! Backends: [`ambercore_embedded::EmbeddedAmberCore`] (the in-process engine —
//! AmberCore compiled into the Phoenix binary, launched with the app),
//! [`ollama::OllamaProvider`] (local Ollama-compatible HTTP servers — Ollama and
//! *remote* AmberCore), [`openai::OpenAiProvider`] (cloud OpenAI-compatible
//! APIs), and [`dispatch::DispatchProvider`] (routes trait calls to whichever
//! backend is active, so the agent runtime + health monitor pick up live backend
//! switches without a restart).

pub mod ambercore_embedded;
pub mod dispatch;
pub mod ollama;
pub mod openai;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// A single message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    /// Text content. Empty string allowed for tool-call-only assistant turns.
    #[serde(default)]
    pub content: String,
    /// Tool calls requested by the assistant (assistant role only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For `role == Tool`: the name of the tool that produced this result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

/// Conversation role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

impl ChatRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "user" => ChatRole::User,
            "assistant" => ChatRole::Assistant,
            "tool" => ChatRole::Tool,
            _ => ChatRole::System,
        }
    }
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: ChatRole::System, content: content.into(), tool_calls: None, tool_name: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: ChatRole::User, content: content.into(), tool_calls: None, tool_name: None }
    }
    pub fn tool_result(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Tool,
            content: content.into(),
            tool_calls: None,
            tool_name: Some(name.into()),
        }
    }
}

/// A tool-call request emitted by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Ollama populates this; some models/flows use it to correlate results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: FunctionRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionRef {
    pub name: String,
    /// Arguments as a JSON object string (per OpenAI/Ollama convention).
    pub arguments: String,
}

/// Tool definition exposed to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub r#type: String, // always "function"
    pub function: ToolFunction,
}

impl ToolDef {
    pub fn function(name: &str, description: &str, parameters: serde_json::Value) -> Self {
        Self {
            r#type: "function".into(),
            function: ToolFunction {
                name: name.to_string(),
                description: description.to_string(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    /// JSON Schema (as a serde_json::Value) describing the parameters.
    pub parameters: serde_json::Value,
}

/// A chat request sent to a model provider.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// Conversation so far, including the system prompt and new user message.
    pub messages: Vec<ChatMessage>,
    /// Tools the model may call.
    pub tools: Vec<ToolDef>,
    /// Sampling temperature.
    pub temperature: f32,
}

/// Usage stats reported after a model turn.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: i64,
    pub output_tokens: i64,
}

/// Live provider stats surfaced to the UI (e.g. the health bar). Backends that
/// don't expose throughput (Ollama) return `None` from [`ModelProvider::stats`].
#[derive(Debug, Clone, Default)]
pub struct ProviderStats {
    /// Last measured generation throughput in tokens/second.
    pub tokens_per_sec: Option<f64>,
}

/// Events streamed from the model as it generates a turn.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// A chunk of assistant text (the visible answer).
    Delta(String),
    /// A chunk of the model's private reasoning / "thinking" (kept separate from
    /// the visible answer). Ollama exposes this as `message.think`; OpenAI-style
    /// providers as `delta.reasoning_content` / `delta.reasoning`; AmberCore
    /// parses Qwen3 `<think>…</think>` out of its text stream.
    Reasoning(String),
    /// The model finished and requested these tool calls (may accompany text).
    ToolCalls(Vec<ToolCall>),
    /// Final usage statistics for the turn.
    Done(Usage),
    /// The provider reported an error mid-stream.
    Error(String),
}

/// A model backend. Implementations stream [`ChatEvent`]s for one turn.
#[async_trait::async_trait]
pub trait ModelProvider: Send + Sync {
    /// List locally available models, if the backend supports discovery.
    async fn list_models(&self) -> Result<Vec<String>>;

    /// Stream one model turn. The returned channel yields [`ChatEvent`]s.
    async fn chat(
        &self,
        model: &str,
        request: ChatRequest,
    ) -> Result<tokio::sync::mpsc::Receiver<ChatEvent>>;

    /// Swap the backend endpoint at runtime (e.g. toggling AmberCore ↔ Ollama).
    /// Default no-op — only the Ollama/AmberCore provider (which share one HTTP
    /// type) needs it. Because the runtime + health monitor hold clones of the
    /// same `Arc<dyn ModelProvider>`, this reaches both automatically.
    async fn set_endpoint(&self, _url: String) {}

    /// Live backend stats for the UI (e.g. tokens/sec for the health bar).
    /// `None` = the backend doesn't expose stats (Ollama). Default `None`.
    async fn stats(&self) -> Option<ProviderStats> {
        None
    }
}
