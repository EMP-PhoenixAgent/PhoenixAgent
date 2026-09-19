//! Live tool-calling protocol check against a real GGUF (ignored by default).
//!
//! Drives [`generate_events`] exactly the way Phoenix Agent does — a system
//! prompt of its shape, a user request that needs a tool, and OpenAI-style
//! tool definitions — and asserts the model emits a STRUCTURED tool call
//! rather than printing the command as text.
//!
//! Run with:
//! ```text
//! PA_TEST_MODELS=<models dir> cargo test --test tool_call_live -- --ignored --nocapture
//! ```
//! The models dir must contain a Qwen3.5 GGUF (auto-discovered by the catalog
//! scan). CPU is fine — the check needs only a short generation.

use ambercore::backend::{resolve_backend, DeviceChoice};
use ambercore::catalog::Catalog;
use ambercore::server::chat::generate_events;
use ambercore::server::protocol as ac;
use ambercore::server::ServerState;

fn phoenix_style_system_prompt() -> String {
    // Mirrors Phoenix's build_system_prompt shape (identity + its own
    // name/description tool list) — the engine must APPEND its tools section
    // to this, not replace it.
    let mut s = String::new();
    s.push_str("You are Phoenix Agent, a fully-local, autonomous coding assistant.\n\n");
    s.push_str("## How to work\n");
    s.push_str("- Investigate before acting. Use `list_dir`, `read_file`, and `grep` to \
                understand the codebase before making changes.\n\n");
    s.push_str("## Tools\n");
    s.push_str("Call a tool by emitting a tool call with the function name and JSON arguments.\n");
    s.push_str("- `list_dir`: List files and folders in a directory.\n");
    s.push_str("- `read_file`: Read a text file's contents.\n");
    s.push_str("- `run_command`: Run a shell command and return its output.\n");
    s
}

fn phoenix_style_tools() -> Vec<ac::ToolDef> {
    vec![
        ac::ToolDef {
            kind: "function".into(),
            function: ac::ToolFunction {
                name: "list_dir".into(),
                description: "List files and folders in a directory.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path to list."}
                    },
                    "required": ["path"]
                }),
            },
        },
        ac::ToolDef {
            kind: "function".into(),
            function: ac::ToolFunction {
                name: "read_file".into(),
                description: "Read a text file's contents.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read."}
                    },
                    "required": ["path"]
                }),
            },
        },
        ac::ToolDef {
            kind: "function".into(),
            function: ac::ToolFunction {
                name: "run_command".into(),
                description: "Run a shell command in the working directory and return its output.".into(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The command to run."}
                    },
                    "required": ["command"]
                }),
            },
        },
    ]
}

#[tokio::test]
#[ignore = "needs a real Qwen3.5 GGUF on disk (PA_TEST_MODELS env var)"]
async fn qwen35_emits_structured_tool_call() {
    let dir = std::env::var("PA_TEST_MODELS").expect("set PA_TEST_MODELS to a models dir");
    let catalog = Catalog::load(std::path::Path::new(&dir)).expect("catalog loads");
    let tags = catalog.tags();
    let tag = tags
        .iter()
        .find(|t| t.to_ascii_lowercase().contains("qwen3.5"))
        .cloned()
        .unwrap_or_else(|| panic!("no qwen3.5 tag in {tags:?}"));
    eprintln!("using model tag: {tag}");

    let backend = resolve_backend(DeviceChoice::Auto).expect("backend");
    let state = ServerState::new(catalog, backend, 1);

    let req = ac::ChatRequest {
        model: tag.clone(),
        messages: vec![
            ac::ChatMessage {
                role: "system".into(),
                content: phoenix_style_system_prompt(),
                tool_calls: None,
                tool: None,
            },
            ac::ChatMessage {
                role: "user".into(),
                content: "Use the list_dir tool to show me what's inside C:\\Users\\Public."
                    .into(),
                tool_calls: None,
                tool: None,
            },
        ],
        tools: phoenix_style_tools(),
        stream: true,
        temperature: Some(0.0),
    };

    let mut rx = generate_events(&state, &req).await.expect("generation starts");
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            ambercore::server::chat::GenEvent::Content(t) => text.push_str(&t),
            ambercore::server::chat::GenEvent::Reasoning(r) => eprintln!("[think] {r}"),
            ambercore::server::chat::GenEvent::ToolCalls(tc) => tool_calls = tc,
            ambercore::server::chat::GenEvent::Done { .. } => break,
            ambercore::server::chat::GenEvent::Error(e) => panic!("engine error: {e}"),
        }
    }

    eprintln!("[content] {text}");
    eprintln!("[tool_calls] {tool_calls:?}");
    assert!(
        !tool_calls.is_empty(),
        "model emitted no structured tool call — content was: {text}"
    );
    assert_eq!(tool_calls[0].function.name, "list_dir");
    // Arguments must be valid JSON with a "path" key.
    let args: serde_json::Value =
        serde_json::from_str(&tool_calls[0].function.arguments).expect("arguments are JSON");
    assert!(args.get("path").is_some(), "arguments: {args}");
}
