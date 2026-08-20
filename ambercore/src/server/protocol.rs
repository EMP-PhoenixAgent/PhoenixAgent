//! Wire types for the Ollama-compatible API — the exact Phoenix contract.
//!
//! Every type and field name here was derived from a direct audit of
//! `phoenix-agent/src/model/ollama.rs` + `config.rs` + `health.rs`. Treat this
//! module as the **specification** of what Phoenix sends and expects.
//!
//! Key invariants (see `ACRoad.md §3` for the full contract):
//! - `model` tags are passed and echoed **verbatim** (no normalization).
//! - `/api/chat` streams **NDJSON** — one JSON object per line, not SSE.
//! - Each non-terminal line carries `message.content` (a **delta**).
//! - The terminal line sets `done: true` and additionally carries
//!   `message.tool_calls` (if any) + `eval_count` + `prompt_eval_count`.
//! - `tool_calls[].function.arguments` is a **JSON object serialized as a string**.
//! - Tool-result messages carry the tool name in a **top-level** `tool` field.

use serde::{Deserialize, Serialize};

// ────────────────────────────── GET /api/tags ──────────────────────────────

/// Response shape for `GET /api/tags`.
///
/// Phoenix reads only each model's `name` and compares it by exact string
/// equality against the active model tag. This endpoint is also Phoenix's sole
/// health probe: HTTP 200 ⇒ server "up"; the active model tag appearing in the
/// list ⇒ model "available".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagsResponse {
    pub models: Vec<TagEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagEntry {
    pub name: String,
}

// ────────────────────────────── POST /api/chat ─────────────────────────────

/// Request body for `POST /api/chat`.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    /// Model tag, passed verbatim (e.g. `qwen2.5-coder:7b`).
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Omitted entirely by Phoenix when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDef>,
    /// Phoenix always sends `true`.
    #[serde(default = "default_true")]
    pub stream: bool,
    /// Always present.
    #[serde(default)]
    pub temperature: Option<f32>,
}

fn default_true() -> bool {
    true
}

/// One message in the chat request. `role: tool` messages carry the tool name
/// in the top-level `tool` field (not nested).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// `"system" | "user" | "assistant" | "tool"` (lowercase).
    pub role: String,
    pub content: String,
    /// Present on assistant messages that produced tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Present only on `role: "tool"` messages — the tool name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool: Option<String>,
}

/// A tool definition Phoenix sends so the model can emit calls to it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// Always `"function"`.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    /// A JSON Schema object.
    pub parameters: serde_json::Value,
}

// ──────────────────────────── NDJSON stream lines ──────────────────────────

/// One line of the NDJSON streaming response.
///
/// - Non-terminal lines: `{ message: { content: "<delta>" } }, done: false`.
/// - Terminal line: sets `done: true` and adds `message.tool_calls` (if any)
///   plus the two token counts.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChunk {
    /// `#[serde(skip_serializing_if = "Option::is_none")]` so absent message is
    /// omitted entirely. Phoenix tolerates a missing `message`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<ChunkMessage>,
    #[serde(default)]
    pub done: bool,
    /// Output token count (terminal line only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<i64>,
    /// Input (prompt) token count (terminal line only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkMessage {
    /// The text **delta** for this chunk (Phoenix accumulates).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The reasoning/"thinking" delta for this chunk — Qwen3 `<think>` content
    /// parsed out of the text stream. Phoenix surfaces it as a separate stream
    /// (collapsible thinking block). Same wire field Ollama uses (`message.think`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<String>,
    /// Tool calls — emitted only on the terminal line.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// A tool call. Emitted in `message.tool_calls` on the terminal line.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Optional; Phoenix does not require it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub function: FunctionRef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionRef {
    pub name: String,
    /// A JSON object serialized as a **string** (OpenAI/Ollama convention).
    pub arguments: String,
}

impl ChatChunk {
    /// Build a non-terminal delta chunk carrying a piece of answer text.
    pub fn delta(text: impl Into<String>) -> Self {
        Self {
            message: Some(ChunkMessage {
                content: Some(text.into()),
                think: None,
                tool_calls: None,
            }),
            done: false,
            eval_count: None,
            prompt_eval_count: None,
        }
    }

    /// Build a non-terminal delta chunk carrying a piece of reasoning/"thinking"
    /// text (Qwen3 `<think>` content). Kept separate from `content` so Phoenix
    /// can render a collapsible thinking block.
    pub fn think(text: impl Into<String>) -> Self {
        Self {
            message: Some(ChunkMessage {
                content: None,
                think: Some(text.into()),
                tool_calls: None,
            }),
            done: false,
            eval_count: None,
            prompt_eval_count: None,
        }
    }

    /// Build the terminal chunk. Carries tool calls + token counts.
    pub fn done(
        tool_calls: Option<Vec<ToolCall>>,
        eval_count: i64,
        prompt_eval_count: i64,
    ) -> Self {
        Self {
            message: Some(ChunkMessage {
                content: None,
                think: None,
                tool_calls,
            }),
            done: true,
            eval_count: Some(eval_count),
            prompt_eval_count: Some(prompt_eval_count),
        }
    }

    /// Serialize to one NDJSON line (no trailing newline).
    pub fn to_ndjson(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_chunk_serializes_to_contract() {
        let chunk = ChatChunk::delta("Hello");
        let json = chunk.to_ndjson().unwrap();
        // Must carry message.content and done:false; must NOT carry counts.
        assert!(json.contains("\"message\":{\"content\":\"Hello\"}"));
        assert!(json.contains("\"done\":false"));
        assert!(!json.contains("eval_count"));
        assert!(!json.contains("prompt_eval_count"));
    }

    #[test]
    fn done_chunk_serializes_to_contract() {
        let chunk = ChatChunk::done(None, 5, 12);
        let json = chunk.to_ndjson().unwrap();
        // Terminal line: done:true + both counts; content omitted.
        assert!(json.contains("\"done\":true"));
        assert!(json.contains("\"eval_count\":5"));
        assert!(json.contains("\"prompt_eval_count\":12"));
        assert!(!json.contains("\"content\""));
    }

    #[test]
    fn tool_call_arguments_must_be_string() {
        // arguments is a *string* per the contract, even though it carries JSON.
        let call = ToolCall {
            id: None,
            function: FunctionRef {
                name: "read_file".into(),
                arguments: r#"{"path":"/etc/hosts"}"#.into(),
            },
        };
        let json = serde_json::to_string(&call).unwrap();
        assert!(json.contains("\"arguments\":\"{\\\"path\\\":\\\"/etc/hosts\\\"}\""));
    }

    #[test]
    fn tool_message_carries_top_level_tool_field() {
        let msg = ChatMessage {
            role: "tool".into(),
            content: "result text".into(),
            tool_calls: None,
            tool: Some("read_file".into()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        // The tool name is at the top level, not nested under tool_calls.
        assert!(json.contains("\"tool\":\"read_file\""));
    }
}
