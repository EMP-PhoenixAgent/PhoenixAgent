//! Tool-calling support: rendering + parsing.
//!
//! Phoenix sends OpenAI-style `tools` definitions and expects structured
//! `tool_calls` back — it does NOT parse tool calls from text. Since candle
//! gives us only raw forward passes (no native function-calling layer), AmberCore
//! implements a **text-protocol shim** using the **Hermes format** that Qwen2.5
//! and Qwen3 models are trained on:
//!
//! - Tools are described in the system prompt, including their JSON Schemas.
//! - The model is told to emit any tool call as:
//!   ```text
//!   <tool_call>
//!   {"name": "tool_name", "arguments": {"key": "value"}}
//!   </tool_call>
//!   ```
//! - AmberCore parses those markers out of the generated text and converts them
//!   into the structured [`ToolCall`] field Phoenix expects, leaving only the
//!   non-tool-call text in the streamed content.
//!
//! This is the standard approach vLLM/Ollama use for Qwen models (the
//! `--tool-call-parser hermes` flag), so Phoenix's existing Qwen2.5-Coder models
//! will work with AmberCore unchanged.

use crate::server::protocol::{FunctionRef, ToolCall, ToolDef};

const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

/// Render the tool-calling instructions + tool definitions for the system prompt.
///
/// Produces a `## Tools` section that (a) describes the Hermes emit format and
/// (b) lists each tool with its full JSON Schema so the model knows the exact
/// argument shapes. Phoenix's own prompt renders only name/description (it
/// carries schemas out-of-band via the API), but since AmberCore is the model
/// server, we render the schema here.
pub fn render_tools_section(tools: &[ToolDef]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut s = String::new();
    s.push_str("## Tools\n");
    s.push_str(
        "You have access to the following tools. To call one, emit ONLY a block wrapped in \
         <tool_call></tool_call> tags containing a JSON object with \"name\" and \"arguments\" \
         fields, and nothing else. Example:\n\
         <tool_call>\n\
         {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n\
         </tool_call>\n\n",
    );
    for tool in tools {
        let f = &tool.function;
        s.push_str(&format!("- {}:", f.name));
        if !f.description.is_empty() {
            s.push_str(&format!(" {}", f.description));
        }
        s.push('\n');
        // Render the parameters schema as pretty JSON so the model sees exact
        // argument names + types.
        if let Ok(pretty) = serde_json::to_string_pretty(&f.parameters) {
            s.push_str(&format!("  parameters:\n{}\n", indent(&pretty, "  ")));
        }
    }
    s
}

fn indent(text: &str, prefix: &str) -> String {
    text.lines()
        .map(|l| format!("{prefix}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A tool call extracted from the model's generated text, plus the surrounding
/// text that should be streamed to the client (everything outside the markers).
#[derive(Debug, Clone)]
pub struct ParsedOutput {
    /// Tool calls found in the text (may be empty).
    pub tool_calls: Vec<ToolCall>,
    /// The text with all `<tool_call>...</tool_call>` blocks removed, for
    /// streaming to the client as content. Whitespace around removed blocks is
    /// trimmed.
    pub remaining_text: String,
}

/// Parse the Hermes `<tool_call>` markers out of generated text.
///
/// Handles multiple calls in one response. Each call's inner JSON must have
/// `name` (string) and `arguments` (object); `arguments` is re-serialized to a
/// string to match Phoenix's wire contract (`FunctionRef.arguments` is a JSON
/// string, not an object). Malformed blocks are dropped (not fatal) — the model
/// may emit partial text that isn't yet a complete call.
pub fn parse_tool_calls(text: &str) -> ParsedOutput {
    let mut tool_calls = Vec::new();
    let mut remaining = String::with_capacity(text.len());
    let mut cursor = 0usize;

    while let Some(open_rel) = text[cursor..].find(TOOL_OPEN) {
        let open_abs = cursor + open_rel;
        // Keep text before the marker.
        remaining.push_str(&text[cursor..open_abs]);

        let after_open = open_abs + TOOL_OPEN.len();
        match text[after_open..].find(TOOL_CLOSE) {
            Some(close_rel) => {
                let close_abs = after_open + close_rel;
                let inner = text[after_open..close_abs].trim();
                if let Some(call) = parse_one_call(inner) {
                    tool_calls.push(call);
                }
                cursor = close_abs + TOOL_CLOSE.len();
            }
            None => {
                // Unclosed marker — the model is mid-generation. Drop the
                // partial marker text (don't stream `<tool_call>` to the
                // client); everything after is consumed.
                cursor = text.len();
                break;
            }
        }
    }
    // Tail text after the last marker.
    remaining.push_str(&text[cursor..]);
    ParsedOutput {
        tool_calls,
        remaining_text: remaining.trim().to_string(),
    }
}

/// Parse one inner JSON object `{"name": ..., "arguments": ...}` into a [`ToolCall`].
fn parse_one_call(inner: &str) -> Option<ToolCall> {
    let v: serde_json::Value = serde_json::from_str(inner).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    // arguments may be an object or already a string; normalize to a JSON string.
    let arguments = match v.get("arguments") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(other) => serde_json::to_string(other).ok()?,
        None => "{}".to_string(),
    };
    Some(ToolCall {
        id: None,
        function: FunctionRef { name, arguments },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::protocol::ToolFunction;
    use serde_json::json;

    fn make_tool(name: &str, desc: &str, params: serde_json::Value) -> ToolDef {
        ToolDef {
            kind: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: desc.into(),
                parameters: params,
            },
        }
    }

    #[test]
    fn render_tools_section_empty_when_no_tools() {
        assert_eq!(render_tools_section(&[]), "");
    }

    #[test]
    fn render_tools_section_includes_name_desc_and_schema() {
        let tools = vec![make_tool(
            "get_weather",
            "Get the weather for a city.",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )];
        let s = render_tools_section(&tools);
        assert!(s.contains("## Tools"));
        assert!(s.contains("- get_weather: Get the weather for a city."));
        assert!(s.contains("\"city\""));
        assert!(s.contains("<tool_call>"));
    }

    #[test]
    fn parse_single_tool_call() {
        let text = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let out = parse_tool_calls(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "get_weather");
        // arguments is a JSON STRING (Phoenix contract), not an object.
        assert_eq!(out.tool_calls[0].function.arguments, "{\"city\":\"Paris\"}");
        assert!(out.remaining_text.is_empty());
    }

    #[test]
    fn parse_tool_call_keeps_surrounding_text() {
        let text = "Let me check that.\n<tool_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>\nDone.";
        let out = parse_tool_calls(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.name, "f");
        assert_eq!(out.tool_calls[0].function.arguments, "{}");
        assert!(out.remaining_text.contains("Let me check that."));
        assert!(out.remaining_text.contains("Done."));
        assert!(!out.remaining_text.contains("<tool_call>"));
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let text = "<tool_call>{\"name\":\"a\",\"arguments\":{\"x\":1}}</tool_call>\n<tool_call>{\"name\":\"b\",\"arguments\":{}}</tool_call>";
        let out = parse_tool_calls(text);
        assert_eq!(out.tool_calls.len(), 2);
        assert_eq!(out.tool_calls[0].function.name, "a");
        assert_eq!(out.tool_calls[1].function.name, "b");
    }

    #[test]
    fn parse_no_markers_returns_text_unchanged() {
        let text = "Just a normal response with no tool calls.";
        let out = parse_tool_calls(text);
        assert!(out.tool_calls.is_empty());
        assert_eq!(out.remaining_text, text);
    }

    #[test]
    fn parse_unclosed_marker_dropped() {
        // The model is mid-generation; the unclosed marker is dropped.
        let text = "thinking... <tool_call>{\"name\":\"a\",\"arguments\":{}}";
        let out = parse_tool_calls(text);
        assert!(out.tool_calls.is_empty());
        assert!(out.remaining_text.contains("thinking"));
        assert!(!out.remaining_text.contains("<tool_call>"));
    }

    #[test]
    fn parse_arguments_as_string_when_model_emits_string() {
        // Some models emit arguments as a JSON string already; normalize it.
        let text = "<tool_call>{\"name\":\"f\",\"arguments\":\"{\\\"k\\\":1}\"}</tool_call>";
        let out = parse_tool_calls(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.arguments, "{\"k\":1}");
    }

    #[test]
    fn parse_missing_arguments_defaults_to_empty_object() {
        let text = "<tool_call>{\"name\":\"f\"}</tool_call>";
        let out = parse_tool_calls(text);
        assert_eq!(out.tool_calls.len(), 1);
        assert_eq!(out.tool_calls[0].function.arguments, "{}");
    }
}
