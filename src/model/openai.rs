//! OpenAI-compatible cloud backend.
//!
//! Talks to any provider implementing the OpenAI Chat Completions wire format
//! (`POST {base}/v1/chat/completions`, `GET {base}/v1/models`) with Bearer auth.
//! This covers OpenAI itself plus OpenRouter, Together, Groq, Anyscale, and
//! local OpenAI-compatible servers (LM Studio, vLLM, Ollama's OpenAI shim).
//!
//! Streaming uses Server-Sent Events (`data: {…}\n\n`, terminated by
//! `data: [DONE]`). Unlike Ollama's NDJSON, OpenAI streams tool-call argument
//! *fragments* across many chunks, so we accumulate them and emit a single
//! [`ChatEvent::ToolCalls`] at the end.

use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::{PhoenixError, Result};
use crate::model::{
    ChatEvent, ChatMessage, ChatRequest, ChatRole, ModelProvider, ToolCall, ToolDef, Usage,
};

/// An OpenAI-compatible cloud provider.
///
/// `base_url` + `api_key` are swappable at runtime (via [`OpenAiProvider::configure`])
/// so the dispatch layer can point the same instance at different registered
/// providers without re-allocating.
pub struct OpenAiProvider {
    inner: tokio::sync::RwLock<CloudEndpoint>,
    client: Client,
}

/// The active cloud endpoint — mutated when the user runs a different provider.
struct CloudEndpoint {
    base_url: String,
    api_key: String,
}

impl OpenAiProvider {
    pub fn new() -> Self {
        Self {
            inner: tokio::sync::RwLock::new(CloudEndpoint {
                base_url: String::new(),
                api_key: String::new(),
            }),
            client: Client::builder()
                .build()
                .expect("failed to build reqwest client"),
        }
    }

    /// Point the provider at a base URL + API key. Reaches the agent runtime +
    /// health monitor automatically (they share the dispatch `Arc`, which holds
    /// this instance).
    pub async fn configure(&self, base_url: String, api_key: String) {
        let mut g = self.inner.write().await;
        g.base_url = base_url.trim_end_matches('/').to_string();
        g.api_key = api_key;
    }

    async fn endpoint(&self) -> (String, String) {
        let g = self.inner.read().await;
        (g.base_url.clone(), g.api_key.clone())
    }
}

impl Default for OpenAiProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelProvider for OpenAiProvider {
    async fn list_models(&self) -> Result<Vec<String>> {
        let (base, key) = self.endpoint().await;
        if base.is_empty() {
            return Err(PhoenixError::Model(
                "no cloud provider configured".to_string(),
            ));
        }
        let url = format!("{base}/v1/models");
        let resp = self
            .client
            .get(&url)
            .bearer_auth(&key)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(PhoenixError::Model(format!(
                "GET {} returned {}",
                url,
                resp.status()
            )));
        }
        let parsed: ModelsResp = resp.json().await?;
        Ok(parsed.data.into_iter().map(|m| m.id).collect())
    }

    async fn chat(
        &self,
        model: &str,
        request: ChatRequest,
    ) -> Result<mpsc::Receiver<ChatEvent>> {
        let (base, key) = self.endpoint().await;
        if base.is_empty() {
            return Err(PhoenixError::Model(
                "no cloud provider configured".to_string(),
            ));
        }
        let body = ChatBody::from_request(model, &request);
        let url = format!("{base}/v1/chat/completions");

        let resp = self
            .client
            .post(&url)
            .bearer_auth(&key)
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
            if let Err(e) = pump_sse(byte_stream, tx.clone()).await {
                let _ = tx.send(ChatEvent::Error(e.to_string())).await;
            }
        });

        Ok(rx)
    }

    /// Cloud providers expose no tok/s endpoint.
    async fn stats(&self) -> Option<crate::model::ProviderStats> {
        None
    }
}

/// Drive the SSE byte stream, emitting events. Each SSE event is one or more
/// `data:` lines terminated by a blank line. We accumulate tool-call argument
/// fragments per index + name and emit a single `ToolCalls` (plus `Done`) when
/// we see `data: [DONE]`.
async fn pump_sse(
    mut stream: impl futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Unpin,
    tx: mpsc::Sender<ChatEvent>,
) -> Result<()> {
    let mut buf = String::new();
    // Accumulate partial tool calls across chunks. Keyed by call index.
    let mut tool_acc: HashMap<i64, PartialToolCall> = HashMap::new();
    let mut final_usage: Option<Usage> = None;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        // SSE events are separated by a blank line. Process each complete event.
        while let Some(blank) = find_event_end(&buf) {
            let event: String = buf.drain(..blank.end).collect();
            // Within an event, gather all `data:` lines (ignoring others).
            for data in extract_data_lines(&event) {
                let data = data.trim();
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    // Emit accumulated tool calls.
                    let mut calls: Vec<ToolCall> = Vec::new();
                    for (_, partial) in std::mem::take(&mut tool_acc).into_iter() {
                        if let Some(c) = partial.finalize() {
                            calls.push(c);
                        }
                    }
                    if !calls.is_empty() {
                        let _ = tx.send(ChatEvent::ToolCalls(calls)).await;
                    }
                    let usage = final_usage.unwrap_or_default();
                    let _ = tx.send(ChatEvent::Done(usage)).await;
                    return Ok(());
                }
                let Ok(parsed) = serde_json::from_str::<StreamChunk>(data) else {
                    continue;
                };

                if let Some(choice) = parsed.choices.first() {
                    // Reasoning/thinking delta (DeepSeek `reasoning_content` or
                    // OpenAI/OpenRouter `reasoning`). Surfaced separately from the
                    // visible answer.
                    if let Some(rc) = choice.delta.reasoning_content.as_deref() {
                        if !rc.is_empty()
                            && tx.send(ChatEvent::Reasoning(rc.to_string())).await.is_err()
                        {
                            return Ok(()); // receiver dropped
                        }
                    }
                    if let Some(r) = choice.delta.reasoning.as_deref() {
                        if !r.is_empty()
                            && tx.send(ChatEvent::Reasoning(r.to_string())).await.is_err()
                        {
                            return Ok(()); // receiver dropped
                        }
                    }
                    // Text delta.
                    if let Some(content) = choice.delta.content.as_deref() {
                        if !content.is_empty()
                            && tx.send(ChatEvent::Delta(content.to_string())).await.is_err()
                        {
                            return Ok(()); // receiver dropped
                        }
                    }
                    // Tool-call fragments.
                    if let Some(parts) = choice.delta.tool_calls.as_ref() {
                        for part in parts {
                            let slot =
                                tool_acc.entry(part.index).or_default();
                            if let Some(name) = part.function.as_ref().and_then(|f| f.name.as_deref())
                            {
                                slot.name = Some(name.to_string());
                            }
                            if let Some(args) =
                                part.function.as_ref().and_then(|f| f.arguments.as_deref())
                            {
                                slot.arguments.push_str(args);
                            }
                        }
                    }
                }

                if let Some(u) = parsed.usage {
                    final_usage = Some(Usage {
                        input_tokens: u.prompt_tokens.unwrap_or(0),
                        output_tokens: u.completion_tokens.unwrap_or(0),
                    });
                }
            }
        }
    }

    // Stream ended without an explicit [DONE] — emit what we have.
    let mut calls: Vec<ToolCall> = Vec::new();
    for (_, partial) in tool_acc.into_iter() {
        if let Some(c) = partial.finalize() {
            calls.push(c);
        }
    }
    if !calls.is_empty() {
        let _ = tx.send(ChatEvent::ToolCalls(calls)).await;
    }
    let _ = tx.send(ChatEvent::Done(final_usage.unwrap_or_default())).await;
    Ok(())
}

/// A tool call being assembled across SSE chunks.
#[derive(Default)]
struct PartialToolCall {
    name: Option<String>,
    arguments: String,
}

impl PartialToolCall {
    fn finalize(self) -> Option<ToolCall> {
        let name = self.name?;
        Some(ToolCall {
            id: None,
            function: crate::model::FunctionRef {
                name,
                arguments: self.arguments,
            },
        })
    }
}

/// Find the next SSE event terminator (a blank line). Returns the byte range to
/// drain (inclusive of the terminator).
fn find_event_end(buf: &str) -> Option<std::ops::Range<usize>> {
    // A blank line is `\n\n` (or `\r\n\r\n`).
    let lf = buf.find("\n\n").map(|i| i + 2)?;
    Some(0..lf)
}

/// Pull every `data:` line out of an SSE event block.
fn extract_data_lines(event: &str) -> Vec<&str> {
    event
        .lines()
        .filter_map(|line| {
            let line = line.strip_prefix('\r').unwrap_or(line);
            line.strip_prefix("data:").map(|d| d.strip_prefix(' ').unwrap_or(d))
        })
        .collect()
}

// ---- Request/response wire types -------------------------------------------

#[derive(Serialize)]
struct ChatBody {
    model: String,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    stream: bool,
    stream_options: StreamOptions,
    temperature: f32,
}

impl ChatBody {
    fn from_request(model: &str, req: &ChatRequest) -> Self {
        let messages = req.messages.iter().map(WireMessage::from_chat).collect();
        let tools = req.tools.iter().map(WireTool::from_def).collect();
        // Let the model decide when to call tools. Omitting tool_choice entirely
        // is equivalent to "auto" for most providers, but being explicit avoids
        // ambiguity on stricter servers.
        let tool_choice = if req.tools.is_empty() {
            None
        } else {
            Some("auto".to_string())
        };
        Self {
            model: model.to_string(),
            messages,
            tools,
            tool_choice,
            stream: true,
            // Ask the server to include final usage in the terminal chunk.
            stream_options: StreamOptions { include_usage: true },
            temperature: req.temperature,
        }
    }
}

/// We request `usage` in the stream so we can record token consumption.
#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct WireMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ToolCall>>,
    /// For `role == "tool"` results: the call id this answers (OpenAI uses the
    /// call id; we don't track ids, so we send the tool name in its place).
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
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
            tool_call_id: m.tool_name.clone(),
        }
    }
}

/// OpenAI-shaped tool definition (same nesting as our [`ToolDef`]).
#[derive(Serialize)]
struct WireTool {
    r#type: String,
    function: ToolDef,
}

impl WireTool {
    fn from_def(d: &ToolDef) -> Self {
        Self {
            r#type: d.r#type.clone(),
            function: d.clone(),
        }
    }
}

#[derive(Deserialize, Debug)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<ChunkUsage>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    #[serde(default)]
    delta: Delta,
}

#[derive(Deserialize, Debug, Default)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    /// DeepSeek-style reasoning stream (also used by some OpenAI-compatible shims).
    #[serde(default)]
    reasoning_content: Option<String>,
    /// OpenAI o-series / OpenRouter-style reasoning stream.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Deserialize, Debug)]
struct DeltaToolCall {
    index: i64,
    #[serde(default)]
    function: Option<DeltaFunction>,
}

#[derive(Deserialize, Debug)]
struct DeltaFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ChunkUsage {
    prompt_tokens: Option<i64>,
    completion_tokens: Option<i64>,
}

#[derive(Deserialize)]
struct ModelsResp {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::stream;

    #[test]
    fn extracts_sse_data_lines() {
        let event = "data: {\"a\":1}\n\ndata: [DONE]\n\n";
        // Simulate two events.
        let mut data = Vec::new();
        data.extend(extract_data_lines("data: {\"a\":1}\n"));
        data.extend(extract_data_lines("data: [DONE]\n"));
        assert_eq!(data, vec!["{\"a\":1}", "[DONE]"]);
        let _ = event; // keep the literal for readability
    }

    /// Feed a fake SSE byte stream through `pump_sse` and collect the events.
    async fn pump(raw: &str) -> Vec<ChatEvent> {
        let item: reqwest::Result<Bytes> = Ok(Bytes::from(raw.to_string()));
        let (tx, mut rx) = mpsc::channel::<ChatEvent>(64);
        pump_sse(stream::iter(vec![item]), tx).await.unwrap();
        let mut out = Vec::new();
        while let Some(e) = rx.recv().await {
            out.push(e);
        }
        out
    }

    #[tokio::test]
    async fn reasoning_content_emits_reasoning() {
        let events = pump(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"}}]}\n\n\
             data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n\
             data: [DONE]\n\n",
        )
        .await;
        assert!(events
            .iter()
            .any(|e| matches!(e, ChatEvent::Reasoning(t) if t == "thinking...")));
        assert!(events
            .iter()
            .any(|e| matches!(e, ChatEvent::Delta(t) if t == "answer")));
    }
}
