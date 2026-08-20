//! Ollama HTTP backend.
//!
//! Talks to the Ollama REST API (`/api/chat`) using streaming NDJSON. Tool
//! calls are passed natively via Ollama's `tools` array and returned in the
//! final JSON object's `message.tool_calls`. We emit text deltas as they
//! arrive and a final [`ChatEvent::ToolCalls`] / [`ChatEvent::Done`].

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::{PhoenixError, Result};
use crate::model::{
    ChatEvent, ChatMessage, ChatRequest, ChatRole, ModelProvider, ProviderStats, ToolCall, Usage,
};

/// Ollama backend.
///
/// The `base_url` is held behind an `RwLock` so it can be swapped at runtime
/// (toggling between Ollama on `:11434` and AmberCore on `:42069`). Because the
/// agent runtime and health monitor each hold a clone of the shared
/// `Arc<dyn ModelProvider>`, flipping the URL here propagates to both
/// automatically — the next `chat()` / `list_models()` call reads the new URL.
pub struct OllamaProvider {
    base_url: tokio::sync::RwLock<String>,
    client: Client,
}

impl OllamaProvider {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: tokio::sync::RwLock::new(
                base_url.into().trim_end_matches('/').to_string(),
            ),
            client: Client::builder()
                .build()
                .expect("failed to build reqwest client"),
        }
    }

    /// Read the current base URL (cloned, so the caller owns it).
    async fn url(&self) -> String {
        self.base_url.read().await.clone()
    }
}

#[async_trait]
impl ModelProvider for OllamaProvider {
    async fn list_models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct TagsResp {
            #[serde(default)]
            models: Vec<TagsModel>,
        }
        #[derive(Deserialize)]
        struct TagsModel {
            name: String,
        }

        let base = self.url().await;
        let url = format!("{base}/api/tags");
        let resp = self.client.get(&url).send().await?;
        if !resp.status().is_success() {
            return Err(PhoenixError::Model(format!(
                "GET {} returned {}",
                url,
                resp.status()
            )));
        }
        let parsed: TagsResp = resp.json().await?;
        Ok(parsed.models.into_iter().map(|m| m.name).collect())
    }

    async fn chat(
        &self,
        model: &str,
        request: ChatRequest,
    ) -> Result<mpsc::Receiver<ChatEvent>> {
        let body = ChatBody::from_request(model, &request);
        let base = self.url().await;
        let url = format!("{base}/api/chat");

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(PhoenixError::Model(format!(
                "POST {} returned {}: {text}",
                url, status
            )));
        }

        let (tx, rx) = mpsc::channel::<ChatEvent>(64);

        let byte_stream = resp.bytes_stream();
        tokio::spawn(async move {
            if let Err(e) = pump_stream(byte_stream, tx.clone()).await {
                let _ = tx.send(ChatEvent::Error(e.to_string())).await;
            }
        });

        Ok(rx)
    }

    /// Swap the base URL at runtime. Reaches the agent runtime + health monitor
    /// automatically (they share this instance via `Arc<dyn ModelProvider>`).
    async fn set_endpoint(&self, url: String) {
        *self.base_url.write().await = url.trim_end_matches('/').to_string();
    }

    /// Probe the backend's `/api/stats` for live throughput. AmberCore serves
    /// this; Ollama does not (returns a non-200 / connection error), so we
    /// degrade gracefully to `None`.
    async fn stats(&self) -> Option<ProviderStats> {
        #[derive(Deserialize)]
        struct StatsResp {
            #[serde(default, rename = "tokens_per_sec")]
            tokens_per_sec: Option<f64>,
        }
        let base = self.url().await;
        let url = format!("{base}/api/stats");
        let resp = self.client.get(&url).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let parsed: StatsResp = resp.json().await.ok()?;
        parsed.tokens_per_sec.map(|tps| ProviderStats {
            tokens_per_sec: Some(tps),
        })
    }
}

/// Drive the NDJSON byte stream, emitting events. Each line is a JSON object
/// whose `message.content` accumulates the assistant text, and whose final
/// object carries `message.tool_calls` and `eval_count`/`prompt_eval_count`.
async fn pump_stream(
    mut stream: impl futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin,
    tx: mpsc::Sender<ChatEvent>,
) -> Result<()> {
    let mut buf = Vec::<u8>::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buf.extend_from_slice(&chunk);

        // Process complete newline-terminated lines.
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parsed: StreamChunk = match serde_json::from_str(line) {
                Ok(p) => p,
                Err(_) => continue, // skip partial / non-JSON lines
            };

            // Emit reasoning/thinking delta if present (kept separate from content).
            if let Some(think) = parsed.message.as_ref().and_then(|m| m.think.clone()) {
                if !think.is_empty()
                    && tx.send(ChatEvent::Reasoning(think)).await.is_err()
                {
                    return Ok(()); // receiver dropped
                }
            }

            // Emit text delta if present.
            if let Some(content) = parsed.message.as_ref().and_then(|m| m.content.clone()) {
                if !content.is_empty() {
                    if tx.send(ChatEvent::Delta(content)).await.is_err() {
                        return Ok(()); // receiver dropped
                    }
                }
            }

            // On final chunk, emit tool calls and usage.
            if parsed.done {
                if let Some(tool_calls) =
                    parsed.message.as_ref().and_then(|m| m.tool_calls.clone())
                {
                    if !tool_calls.is_empty() {
                        let _ = tx.send(ChatEvent::ToolCalls(tool_calls)).await;
                    }
                }
                let usage = Usage {
                    input_tokens: parsed.prompt_eval_count.unwrap_or(0),
                    output_tokens: parsed.eval_count.unwrap_or(0),
                };
                let _ = tx.send(ChatEvent::Done(usage)).await;
            }
        }
    }
    Ok(())
}

// ---- Request/response wire types ------------------------------------------

#[derive(Serialize)]
struct ChatBody {
    model: String,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<crate::model::ToolDef>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

impl ChatBody {
    fn from_request(model: &str, req: &ChatRequest) -> Self {
        let messages = req
            .messages
            .iter()
            .map(WireMessage::from_chat)
            .collect();
        Self {
            model: model.to_string(),
            messages,
            tools: req.tools.clone(),
            stream: true,
            temperature: Some(req.temperature),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct WireMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool: Option<String>, // tool name, for role=tool results
}

impl WireMessage {
    fn from_chat(m: &ChatMessage) -> Self {
        let role = match m.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        }
        .to_string();
        Self {
            role,
            content: if m.content.is_empty() { None } else { Some(m.content.clone()) },
            tool_calls: m.tool_calls.clone(),
            tool: m.tool_name.clone(),
        }
    }
}

#[derive(Deserialize, Debug)]
struct StreamChunk {
    #[serde(default)]
    message: Option<ChunkMessage>,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    eval_count: Option<i64>,
    #[serde(default)]
    prompt_eval_count: Option<i64>,
}

#[derive(Deserialize, Debug)]
struct ChunkMessage {
    #[serde(default)]
    content: Option<String>,
    /// Ollama's reasoning/"thinking" field (Qwen3, DeepSeek-R1, etc. populate
    /// `message.think`). Surfaced as a separate reasoning stream. AmberCore
    /// reuses this same wire field for its parsed `<think>` content.
    #[serde(default)]
    think: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;

    /// Feed a fake NDJSON stream through `pump_stream` and collect the events.
    async fn pump_lines(lines: &[&str]) -> Vec<ChatEvent> {
        let body = lines.iter().map(|l| format!("{l}\n")).collect::<String>();
        let item: reqwest::Result<Bytes> = Ok(Bytes::from(body));
        let (tx, mut rx) = mpsc::channel::<ChatEvent>(64);
        pump_stream(stream::iter(vec![item]), tx).await.unwrap();
        let mut out = Vec::new();
        while let Some(e) = rx.recv().await {
            out.push(e);
        }
        out
    }

    #[tokio::test]
    async fn think_field_emits_reasoning_then_delta() {
        let events = pump_lines(&[
            r#"{"message":{"think":"why","content":"ans"},"done":false}"#,
        ])
        .await;
        assert!(events
            .iter()
            .any(|e| matches!(e, ChatEvent::Reasoning(t) if t == "why")));
        assert!(events
            .iter()
            .any(|e| matches!(e, ChatEvent::Delta(t) if t == "ans")));
    }

    #[tokio::test]
    async fn done_chunk_carries_usage() {
        let events = pump_lines(&[r#"{"done":true,"eval_count":7,"prompt_eval_count":3}"#]).await;
        assert!(events
            .iter()
            .any(|e| matches!(e, ChatEvent::Done(u) if u.output_tokens == 7 && u.input_tokens == 3)));
    }
}
