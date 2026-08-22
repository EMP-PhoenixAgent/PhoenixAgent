//! Tool registry and the built-in toolset.
//!
//! A [`Tool`] is a unit of capability the agent can invoke. The registry
//! exposes tool definitions to the model and dispatches calls through an
//! approval gate (see [`crate::config::ApprovalPolicy`]). All built-in tools
//! live here; the registry is a thin wrapper.

pub mod fs;
pub mod search;
pub mod shell;
pub mod user_script;

use async_trait::async_trait;
use serde_json::Value;

use crate::config::{ApprovalPolicy, ToolKind};
use crate::error::{PhoenixError, Result};
use crate::model::ToolDef;

/// Context passed to every tool invocation.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Working directory the agent operates in (project root).
    pub workdir: std::path::PathBuf,
    /// Operating system string for diagnostics.
    pub os: String,
}

/// The outcome of running a tool.
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// Whether the tool succeeded.
    pub success: bool,
    /// The textual content returned to the model.
    pub content: String,
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self { success: true, content: content.into() }
    }
    pub fn err(content: impl Into<String>) -> Self {
        Self { success: false, content: content.into() }
    }
}

/// A unit of capability the agent can invoke.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Machine name, e.g. `read_file`.
    fn name(&self) -> &str;
    /// Human/model-facing description.
    fn description(&self) -> &str;
    /// JSON Schema describing the parameters object.
    fn parameters_schema(&self) -> Value;
    /// Coarse effect classification for the approval gate.
    fn kind(&self) -> ToolKind;
    /// Execute the tool.
    async fn run(&self, args: &Value, ctx: &ToolContext) -> ToolResult;
}

/// Builds a [`ToolDef`] from a [`Tool`] for exposure to the model.
pub fn tool_def<T: Tool + ?Sized>(t: &T) -> ToolDef {
    ToolDef::function(t.name(), t.description(), t.parameters_schema())
}

/// Build `Box<dyn Tool>` objects for a set of tool rows (the enabled user
/// tools for the active profile). Invalid JSON schemas fall back to an empty
/// object so a malformed tool is still callable (it just declares no params).
pub fn build_user_tools(tools: &[crate::db::ToolRow]) -> Vec<Box<dyn Tool>> {
    tools
        .iter()
        .map(|t| {
            let schema: Value = serde_json::from_str(&t.params_schema)
                .unwrap_or_else(|_| Value::Object(Default::default()));
            let kind = match t.tool_kind.as_str() {
                "read" => ToolKind::Read,
                _ => ToolKind::Write,
            };
            Box::new(user_script::UserScriptTool::new(
                &t.name,
                &t.description,
                &t.interpreter,
                &t.script_body,
                schema,
                kind,
            )) as Box<dyn Tool>
        })
        .collect()
}

/// The `delegate` tool — lets the main agent hand a specialist sub-task to a
/// predefined sub-agent (Panel 6). This built-in only ADVERTISES the tool to the
/// model; execution is intercepted by the runtime (`run_delegation`), which runs
/// the named sub-agent on its configured model via a nested chat. The runtime
/// also rewrites this tool's description per turn to name the available
/// sub-agents.
pub struct DelegateTool;

#[async_trait]
impl Tool for DelegateTool {
    fn name(&self) -> &str {
        "delegate"
    }
    fn description(&self) -> &str {
        "Delegate a specialist sub-task to a predefined sub-agent. Call with \
         {\"sub_agent\":\"<name>\",\"task\":\"<what to do>\"}. (The runtime fills \
         in the list of available sub-agents.)"
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "sub_agent": {"type": "string", "description": "Name of the sub-agent to delegate to."},
                "task": {"type": "string", "description": "The sub-task to perform."}
            },
            "required": ["sub_agent", "task"]
        })
    }
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
    async fn run(&self, _args: &Value, _ctx: &ToolContext) -> ToolResult {
        // Execution is intercepted by the runtime before this is reached.
        ToolResult::err("delegate is handled by the runtime.")
    }
}

/// The `update_todo` tool — replaces the to-do list shown in the chat's
/// to-do panel. This built-in only ADVERTISES the tool to the model; execution
/// is intercepted by the runtime (`run_update_todo`), which persists the
/// markdown, injects it into the system prompt, and notifies the UI.
pub struct UpdateTodoTool;

#[async_trait]
impl Tool for UpdateTodoTool {
    fn name(&self) -> &str {
        "update_todo"
    }
    fn description(&self) -> &str {
        "Replace the shared to-do list shown to the user in the chat's plan panel. \
         Call with {\"markdown\":\"<full updated markdown>\"} — headings for phases, \
         `- [ ] task` for pending, `- [x] ~~task~~` (strikethrough) for done. Send the \
         COMPLETE list every time (it replaces the old one). Use it whenever a plan \
         is agreed and whenever a task completes."
    }
    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "markdown": {"type": "string", "description": "The full to-do list markdown."}
            },
            "required": ["markdown"]
        })
    }
    fn kind(&self) -> ToolKind {
        // Only touches the in-app plan panel — no filesystem effect.
        ToolKind::Read
    }
    async fn run(&self, _args: &Value, _ctx: &ToolContext) -> ToolResult {
        // Execution is intercepted by the runtime before this is reached.
        ToolResult::err("update_todo is handled by the runtime.")
    }
}

/// The registry of available tools.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
    approval_policy: std::sync::Mutex<ApprovalPolicy>,
}

impl ToolRegistry {
    /// Build the default v0.1 registry with all built-in tools.
    pub fn default_tools(approval_policy: ApprovalPolicy) -> Self {
        Self::default_tools_with(approval_policy, Vec::new())
    }

    /// Build the registry with all built-in tools PLUS the given user tools
    /// (appended after the built-ins).
    pub fn default_tools_with(
        approval_policy: ApprovalPolicy,
        extra: Vec<Box<dyn Tool>>,
    ) -> Self {
        let mut tools: Vec<Box<dyn Tool>> = vec![
            Box::new(fs::ReadFile),
            Box::new(fs::WriteFile),
            Box::new(fs::EditFile),
            Box::new(fs::ListDir),
            Box::new(search::Grep),
            Box::new(shell::RunCommand),
            Box::new(DelegateTool),
            Box::new(UpdateTodoTool),
        ];
        tools.extend(extra);
        Self { tools, approval_policy: std::sync::Mutex::new(approval_policy) }
    }

    /// Expose all tool definitions to the model.
    pub fn definitions(&self) -> Vec<ToolDef> {
        self.tools.iter().map(|t| tool_def(t.as_ref())).collect()
    }

    /// Find a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }

    /// Whether this tool requires user approval under the current policy.
    pub fn requires_approval(&self, name: &str) -> Result<bool> {
        let kind = self
            .get(name)
            .ok_or_else(|| PhoenixError::Tool {
                tool: name.to_string(),
                message: "unknown tool".into(),
            })?
            .kind();
        Ok(self.approval_policy.lock().unwrap().requires_approval(kind))
    }

    /// Live-switch the approval policy without rebuilding the registry (so all
    /// tools — including MCP tools — are preserved). Used by the mode selector.
    pub fn set_policy(&self, policy: ApprovalPolicy) {
        *self.approval_policy.lock().unwrap() = policy;
    }

    /// Dispatch a tool call by name.
    pub async fn execute(&self, name: &str, args: &Value, ctx: &ToolContext) -> Result<ToolResult> {
        let tool = self
            .get(name)
            .ok_or_else(|| PhoenixError::Tool {
                tool: name.to_string(),
                message: "unknown tool".into(),
            })?;
        Ok(tool.run(args, ctx).await)
    }
}

/// Parse the `arguments` JSON string a model emits into a [`Value`].
pub fn parse_arguments(raw: &str) -> Result<Value> {
    if raw.trim().is_empty() {
        return Ok(Value::Object(Default::default()));
    }
    serde_json::from_str(raw)
        .map_err(|e| PhoenixError::Tool {
            tool: "<parse>".into(),
            message: format!("invalid tool arguments JSON: {e}"),
        })
}

/// Fetch a string field from a JSON object argument, with a default.
pub fn arg_str(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
}

/// Fetch a boolean field from a JSON object argument.
pub fn arg_bool(args: &Value, key: &str, default: bool) -> bool {
    args.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

/// Fetch a number field from a JSON object argument.
pub fn arg_u32(args: &Value, key: &str, default: u32) -> u32 {
    args.get(key)
        .and_then(|v| v.as_u64())
        .map(|n| n as u32)
        .unwrap_or(default)
}
