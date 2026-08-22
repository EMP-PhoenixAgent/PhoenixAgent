//! The agent ReAct loop and its handle.
//!
//! Flow per turn:
//! 1. The TUI sends a [`Command::UserInput`] with the user's message.
//! 2. The loop appends it to the conversation, persists it, then calls the model.
//! 3. Model text streams back as [`AgentEvent::AssistantDelta`].
//! 4. If the model requests tool calls:
//!    - Read-only tools run immediately.
//!    - Write tools emit [`AgentEvent::ToolNeedsApproval`]; the TUI must reply
//!      with [`Command::Approve`] or [`Command::Deny`] before the loop resumes.
//! 5. Tool results are persisted and fed back; the loop calls the model again.
//! 6. The loop stops when the model emits no tool calls (final answer) or the
//!    iteration cap is reached.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, Mutex};

use crate::config::{ApprovalPolicy, Config};
use crate::error::Result;
use serde::Serialize;

use crate::model::{ChatEvent, ChatMessage, ChatRequest, ChatRole, ModelProvider, ToolCall};

use super::prompt;
use super::tools::{build_user_tools, parse_arguments, ToolContext, ToolRegistry};
use super::mcp::{self, McpTransport};

/// A serializable description of an enabled user tool, carried across the
/// command channel so the runtime can rebuild the registry without DB access.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub interpreter: String,
    pub script_body: String,
    pub params_schema: String,
    pub tool_kind: String,
}

impl ToolSpec {
    /// Build the specs for a set of tool rows (the enabled user tools).
    pub fn from_rows(rows: &[crate::db::ToolRow]) -> Vec<ToolSpec> {
        rows.iter()
            .map(|t| ToolSpec {
                name: t.name.clone(),
                description: t.description.clone(),
                interpreter: t.interpreter.clone(),
                script_body: t.script_body.clone(),
                params_schema: t.params_schema.clone(),
                tool_kind: t.tool_kind.clone(),
            })
            .collect()
    }

    /// Convert specs back into tool rows for `build_user_tools`.
    fn to_rows(&self) -> Vec<crate::db::ToolRow> {
        // `build_user_tools` only reads the fields below; the others are
        // filled with harmless defaults.
        std::iter::once(crate::db::ToolRow {
            id: 0,
            name: self.name.clone(),
            description: self.description.clone(),
            interpreter: self.interpreter.clone(),
            script_body: self.script_body.clone(),
            params_schema: self.params_schema.clone(),
            tool_kind: self.tool_kind.clone(),
            source: "local".into(),
            source_url: None,
            enabled_global: true,
            created_at: String::new(),
            updated_at: String::new(),
        })
        .collect()
    }
}

/// A serializable snapshot of an enabled MCP connection, carried across the
/// command channel so the runtime can reconnect without DB access.
#[derive(Debug, Clone, Serialize)]
pub struct McpConnectionSpec {
    pub id: i64,
    pub name: String,
    /// `"stdio"` or `"http"`.
    pub transport: String,
    /// Executable (stdio) or base URL (http).
    pub command: String,
    /// JSON array of string args (stdio) / extra config, verbatim.
    pub args_json: String,
}

impl McpConnectionSpec {
    /// Parse the spec into a transport. Falls back to a stdio spawn with the
    /// raw command (so a misconfigured row yields a connect error rather than
    /// a panic).
    pub fn to_transport(&self) -> McpTransport {
        match self.transport.as_str() {
            "http" => McpTransport::Http { url: self.command.clone() },
            _ => {
                let args: Vec<String> = serde_json::from_str(&self.args_json).unwrap_or_default();
                McpTransport::Stdio {
                    command: self.command.clone(),
                    args,
                }
            }
        }
    }
}

/// Commands the TUI sends to the agent loop.
pub enum Command {
    /// New user message for the current session.
    UserInput { text: String },
    /// Approve a pending tool call (identified by the call index).
    Approve { index: usize },
    /// Deny a pending tool call.
    Deny { index: usize },
    /// Reset the conversation to a fresh session.
    NewSession,
    /// Stop/abort the current turn if possible.
    Stop,
    /// Switch the active model live (applies from the next model turn).
    SetModel { model: String },
    /// Switch the reasoning mode (Plan/Think/Auto) live — applies the mode's
    /// approval-policy preset to the registry.
    SetMode { mode: crate::config::Mode },
    /// Replace the to-do list markdown (the shared plan shown in the chat's
    /// to-do panel). Sent by the UI editor; the `update_todo` tool has the
    /// same effect from the agent side. Rebuilds the system prompt so the
    /// model always sees the current plan.
    SetTodo { markdown: String },
    /// Apply a profile's behavior settings live (approval policy, iteration
    /// cap, context window). The tool registry is rebuilt with the new policy.
    ApplyProfileSettings {
        approval_policy: ApprovalPolicy,
        max_iterations: u32,
        context_window: u32,
    },
    /// Change the working directory live (applies from the next user turn).
    SetWorkdir { workdir: PathBuf },
    /// Reload the enabled skills and rebuild the system prompt (`messages[0]`).
    /// Each entry is `(name, description, markdown body)`.
    ReloadSkills { skills: Vec<(String, String, String)> },
    /// Reload the enabled user tools: rebuild the `ToolRegistry` (built-ins +
    /// user scripts) and refresh the system prompt's `## Tools` section.
    ReloadTools { tools: Vec<ToolSpec> },
    /// Reload the enabled context files and rebuild the system prompt's
    /// `## Project Context` section. Each entry is `(name, description, body)`.
    ReloadContext { context: Vec<(String, String, String)> },
    /// Reconnect to the enabled MCP servers and rebuild the tool registry with
    /// their tools (in addition to built-ins + user-script tools), then refresh
    /// the system prompt's `## Tools` section.
    ReloadMcp { connections: Vec<McpConnectionSpec> },
}

/// Events the agent loop emits to the TUI / web frontend.
///
/// Wire format: `#[serde(tag = "type")]` → each variant serializes as
/// `{"type": "<snake_case>", ...named fields}`. All variants use named fields so
/// the frontend can read payloads by key (no serde `"0"` newtype quirk).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// Streaming assistant text chunk (the visible answer).
    AssistantDelta { delta: String },
    /// Streaming assistant reasoning / "thinking" chunk. Rendered in a separate,
    /// collapsible thinking block — never mixed into the answer.
    AssistantReasoning { delta: String },
    /// A full assistant message finished (text complete, optionally with tool
    /// calls). `reasoning` carries the full thinking text, if any, so the UI can
    /// finalize its thinking block.
    AssistantMessage {
        text: String,
        tool_calls: Vec<ToolCall>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    /// A tool is about to run (read-only tools run without approval). `args` is
    /// the full JSON-string arguments (the UI truncates for the header and
    /// pretty-prints in the expanded body).
    ToolStarted {
        index: usize,
        name: String,
        args: String,
    },
    /// A read-only or approved tool finished. `duration_ms` is the wall-clock
    /// elapsed since the matching `ToolStarted`, when measurable.
    ToolFinished {
        index: usize,
        name: String,
        success: bool,
        result: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
    /// A write tool is waiting for user approval.
    ToolNeedsApproval {
        index: usize,
        name: String,
        args: String,
    },
    /// The user denied a tool call.
    ToolDenied { index: usize, name: String },
    /// A sub-agent was invoked via the `delegate` tool (`index` matches the
    /// `delegate` tool call). Its streamed output arrives as `SubAgentDelta` /
    /// `SubAgentReasoning`; the UI renders it nested inside the tool card.
    SubAgentStarted {
        index: usize,
        name: String,
        model: String,
        task: String,
    },
    /// A chunk of a running sub-agent's answer text.
    SubAgentDelta { index: usize, name: String, text: String },
    /// A chunk of a running sub-agent's reasoning / thinking text.
    SubAgentReasoning { index: usize, name: String, text: String },
    /// A delegated sub-agent finished. `result` is the final text it produced.
    SubAgentFinished {
        index: usize,
        name: String,
        model: String,
        result: String,
    },
    /// A turn finished (final answer delivered or iteration cap hit).
    TurnDone { iterations: u32 },
    /// The to-do list (shared plan markdown) changed — the chat's to-do panel
    /// re-renders from this. Emitted by the `update_todo` tool.
    TodoUpdated { markdown: String },
    /// The agent encountered an error.
    Error { message: String },
    /// Status / diagnostic text (not shown as model output).
    Status { message: String },
}

/// Shared runtime state holding the conversation and session bookkeeping, plus
/// the live-mutable workbench settings (model, workdir, behavior profile).
struct Shared {
    /// Full conversation including the system prompt.
    messages: Vec<ChatMessage>,
    /// The active session id (None until first message).
    session_id: Option<i64>,
    /// Pending approval: tool-call index awaiting a decision.
    pending: Option<PendingApproval>,
    /// The currently active model (switchable live via `Command::SetModel`).
    model: String,
    /// The current working directory (switchable live via `Command::SetWorkdir`).
    workdir: PathBuf,
    /// Approval policy for the active profile (switchable live).
    approval_policy: ApprovalPolicy,
    /// Current reasoning mode (Plan/Think/Auto) — drives the approval-policy
    /// preset chosen from the chat mode bar. Defaults to Auto.
    mode: crate::config::Mode,
    /// Max reasoning iterations for the active profile.
    max_iterations: u32,
    /// Context-window size (messages retained per turn) for the active profile.
    context_window: u32,
    /// Enabled skills (name, description, markdown body) for the active profile.
    skills: Vec<(String, String, String)>,
    /// Enabled user tools for the active profile (so NewSession / prompt
    /// rebuilds stay consistent with the registry).
    enabled_user_tools: Vec<ToolSpec>,
    /// Enabled context files (name, description, markdown body) for the active
    /// profile.
    context: Vec<(String, String, String)>,
    /// Enabled MCP connections for the active profile (so NewSession /
    /// ApplyProfileSettings rebuilds stay consistent with the registry).
    mcp: Vec<McpConnectionSpec>,
    /// The current to-do list markdown (the shared plan shown in the chat's
    /// to-do panel). Rendered into the system prompt so the model works from
    /// the same plan the user sees.
    todo: String,
}

struct PendingApproval {
    index: usize,
    call: ToolCall,
    responder: oneshot::Sender<bool>,
}

/// Handle returned to the TUI. Not cloneable (the event receiver is single-consumer).
pub struct AgentHandle {
    pub cmd_tx: mpsc::Sender<Command>,
    pub event_rx: mpsc::Receiver<AgentEvent>,
}

/// Construct and spawn the agent runtime. Returns the handle.
pub struct AgentRuntime;

impl AgentRuntime {
    pub fn spawn(
        provider: Arc<dyn ModelProvider>,
        model: String,
        config: Config,
        workdir: PathBuf,
        store: Arc<Mutex<crate::db::MemoryStore>>,
        skills: Vec<(String, String, String)>,
        enabled_user_tools: Vec<ToolSpec>,
        context: Vec<(String, String, String)>,
        mcp: Vec<McpConnectionSpec>,
        todo: String,
    ) -> AgentHandle {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Command>(16);
        let (event_tx, event_rx) = mpsc::channel::<AgentEvent>(64);

        let ctx = ToolContext {
            workdir: workdir.clone(),
            os: std::env::consts::OS.to_string(),
        };
        // Built-in tools + the profile's enabled user tools.
        let extra = build_user_tools(
            &enabled_user_tools.iter().flat_map(|s| s.to_rows()).collect::<Vec<_>>(),
        );
        let registry = Arc::new(ToolRegistry::default_tools_with(
            config.approval_policy,
            extra,
        ));
        // Collect owned tool summaries (name, description) for the system prompt.
        let tool_summaries_owned: Vec<(String, String)> = registry
            .definitions()
            .into_iter()
            .map(|d| (d.function.name, d.function.description))
            .collect();
        let system_prompt = {
            let refs: Vec<(&str, &str)> = tool_summaries_owned
                .iter()
                .map(|(a, b)| (a.as_str(), b.as_str()))
                .collect();
            let skill_refs: Vec<(&str, &str, &str)> = skills
                .iter()
                .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                .collect();
            let context_refs: Vec<(&str, &str, &str)> = context
                .iter()
                .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                .collect();
            prompt::build_system_prompt(&ctx, &refs, &skill_refs, &context_refs, &todo, crate::config::Mode::default())
        };

        let shared = Arc::new(Mutex::new(Shared {
            messages: vec![ChatMessage::system(system_prompt.clone())],
            session_id: None,
            pending: None,
            model,
            workdir,
            approval_policy: config.approval_policy,
            mode: crate::config::Mode::default(),
            max_iterations: config.max_iterations,
            context_window: config.context_window,
            skills,
            enabled_user_tools,
            context,
            mcp,
            todo,
        }));

        tokio::spawn(run_loop(
            provider,
            registry,
            store,
            shared,
            cmd_rx,
            event_tx,
        ));

        AgentHandle { cmd_tx, event_rx }
    }
}

async fn run_loop(
    provider: Arc<dyn ModelProvider>,
    mut registry: Arc<ToolRegistry>,
    store: Arc<Mutex<crate::db::MemoryStore>>,
    shared: Arc<Mutex<Shared>>,
    mut cmd_rx: mpsc::Receiver<Command>,
    event_tx: mpsc::Sender<AgentEvent>,
) {
    // ---- startup: connect any enabled MCP servers ----------------------
    // The initial registry (built in `spawn`) has no MCP tools; connect them
    // now and rebuild once before processing the first command. Failures are
    // non-fatal (a bad server is logged and skipped).
    {
        let specs = shared.lock().await.mcp.clone();
        if !specs.is_empty() {
            let (policy, workdir) = {
                let s = shared.lock().await;
                (s.approval_policy, s.workdir.clone())
            };
            let extra = connect_mcp_and_user_tools(&specs, &shared, policy).await;
            registry = Arc::new(ToolRegistry::default_tools_with(policy, extra));
            let _ = event_tx
                .send(AgentEvent::Status { message: "MCP connections established at startup.".into() })
                .await;
            // Rebuild the system prompt so the new tool list is visible.
            rebuild_prompt_from(&shared, &registry, &workdir).await;
        }
    }

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            Command::NewSession => {
                let mut s = shared.lock().await;
                // Keep the system prompt (messages[0]); drop the rest.
                let sys = s.messages.first().cloned().unwrap_or_else(|| {
                    let refs: Vec<(&str, &str)> = Vec::new();
                    let skill_refs: Vec<(&str, &str, &str)> = s
                        .skills
                        .iter()
                        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                        .collect();
                    let context_refs: Vec<(&str, &str, &str)> = s
                        .context
                        .iter()
                        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                        .collect();
                    let ctx = ToolContext {
                        workdir: s.workdir.clone(),
                        os: std::env::consts::OS.to_string(),
                    };
                    ChatMessage::system(prompt::build_system_prompt(
                        &ctx, &refs, &skill_refs, &context_refs, &s.todo, s.mode,
                    ))
                });
                s.messages = vec![sys];
                s.session_id = None;
                s.pending = None;
                let _ = event_tx.send(AgentEvent::Status { message: "New session started.".into() }).await;
            }
            Command::UserInput { text } => {
                if let Err(e) =
                    handle_user_input(&provider, &registry, &store, &shared, &event_tx, text).await
                {
                    let _ = event_tx.send(AgentEvent::Error { message: e.to_string() }).await;
                }
            }
            Command::Approve { index } => {
                let responder = {
                    let mut s = shared.lock().await;
                    if let Some(p) = s.pending.take() {
                        if p.index == index {
                            Some(p.responder)
                        } else {
                            // Put it back; mismatched index.
                            s.pending = Some(p);
                            None
                        }
                    } else {
                        None
                    }
                };
                if let Some(r) = responder {
                    let _ = r.send(true);
                }
            }
            Command::Deny { index } => {
                let (responder, call) = {
                    let mut s = shared.lock().await;
                    if let Some(p) = s.pending.take() {
                        if p.index == index {
                            (Some(p.responder), Some(p.call))
                        } else {
                            s.pending = Some(p);
                            (None, None)
                        }
                    } else {
                        (None, None)
                    }
                };
                if let (Some(r), Some(call)) = (responder, call) {
                    let _ = r.send(false);
                    let _ = event_tx
                        .send(AgentEvent::ToolDenied {
                            index,
                            name: call.function.name.clone(),
                        })
                        .await;
                    // Append a tool result indicating denial so the model adapts.
                    let result_msg = ChatMessage::tool_result(
                        call.function.name.clone(),
                        "The user denied this tool call. Do not retry it unless they ask.",
                    );
                    let mut s = shared.lock().await;
                    persist_message(&store, &mut s, &result_msg, None, None, None).await;
                    s.messages.push(result_msg);
                }
            }
            Command::Stop => {
                let _ = event_tx.send(AgentEvent::Status { message: "Stop requested (turn will end at next checkpoint).".into() }).await;
            }
            Command::SetModel { model } => {
                let mut s = shared.lock().await;
                s.model = model.clone();
                let _ = event_tx
                    .send(AgentEvent::Status { message: format!("Model switched to {model}.") })
                    .await;
            }
            Command::SetWorkdir { workdir } => {
                let mut s = shared.lock().await;
                s.workdir = workdir.clone();
                let _ = event_tx
                    .send(AgentEvent::Status {
                        message: format!("Working directory set to {}.", workdir.display()),
                    })
                    .await;
            }
            Command::ApplyProfileSettings {
                approval_policy,
                max_iterations,
                context_window,
            } => {
                // Safe to mutate `registry` here: handle_user_input (the only
                // reader of `&registry`) has fully returned before we reach the
                // top of this loop again.
                registry = Arc::new(ToolRegistry::default_tools(approval_policy));
                let policy_label = match approval_policy {
                    ApprovalPolicy::All => "all",
                    ApprovalPolicy::WritesOnly => "writes_only",
                    ApprovalPolicy::Never => "never",
                };
                {
                    let mut s = shared.lock().await;
                    s.approval_policy = approval_policy;
                    s.max_iterations = max_iterations;
                    s.context_window = context_window;
                }
                let _ = event_tx
                    .send(AgentEvent::Status {
                        message: format!(
                            "Profile applied: approval={policy_label}, max_iterations={max_iterations}, context_window={context_window}."
                        ),
                    })
                    .await;
            }
            Command::SetMode { mode } => {
                let policy = mode.approval_policy();
                {
                    let mut s = shared.lock().await;
                    s.mode = mode;
                    s.approval_policy = policy;
                }
                registry.set_policy(policy);
                // Rebuild the system prompt so the mode directive updates immediately.
                let workdir = shared.lock().await.workdir.clone();
                rebuild_prompt_from(&shared, &registry, &workdir).await;
                let _ = event_tx
                    .send(AgentEvent::Status { message: format!("Mode switched to {}.", mode.label()) })
                    .await;
            }
            Command::SetTodo { markdown } => {
                // Update Shared + rebuild the prompt so the model works from
                // the same plan the user sees. Persistence is the caller's
                // job (the `set_todo` command already saved it to the DB).
                {
                    let mut s = shared.lock().await;
                    s.todo = markdown;
                }
                let workdir = shared.lock().await.workdir.clone();
                rebuild_prompt_from(&shared, &registry, &workdir).await;
            }
            Command::ReloadSkills { skills } => {
                let count = skills.len();
                // Rebuild the system prompt (messages[0]) with the new skill set,
                // preserving the rest of the conversation. We also update
                // Shared.skills so NewSession rebuilds with the current set.
                let tool_refs: Vec<(String, String)> = {
                    let owned = registry.definitions();
                    owned
                        .into_iter()
                        .map(|d| (d.function.name, d.function.description))
                        .collect()
                };
                let mut s = shared.lock().await;
                s.skills = skills.clone();
                let skill_refs: Vec<(&str, &str, &str)> = skills
                    .iter()
                    .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                    .collect();
                let context_refs: Vec<(&str, &str, &str)> = s
                    .context
                    .iter()
                    .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                    .collect();
                let refs: Vec<(&str, &str)> = tool_refs
                    .iter()
                    .map(|(a, b)| (a.as_str(), b.as_str()))
                    .collect();
                let ctx = ToolContext {
                    workdir: s.workdir.clone(),
                    os: std::env::consts::OS.to_string(),
                };
                let new_prompt = prompt::build_system_prompt(
                    &ctx, &refs, &skill_refs, &context_refs, &s.todo, s.mode,
                );
                // Replace the system message in place.
                if let Some(first) = s.messages.first_mut() {
                    first.content = new_prompt;
                } else {
                    s.messages.insert(0, ChatMessage::system(new_prompt));
                }
                drop(s);
                let _ = event_tx
                    .send(AgentEvent::Status {
                        message: format!("Skills reloaded ({count} enabled)."),
                    })
                    .await;
            }
            Command::ReloadTools { tools } => {
                let count = tools.len();
                // Rebuild the registry: built-ins + the new user-tool set. The
                // safety reasoning is the same as ApplyProfileSettings —
                // handle_user_input (the only reader of `&registry`) has fully
                // returned before we reach the top of this loop again.
                let rows = tools.iter().flat_map(|s| s.to_rows()).collect::<Vec<_>>();
                let extra = build_user_tools(&rows);
                // We need the approval policy from Shared to rebuild.
                let (policy, workdir, skills, context, todo, mode) = {
                    let s = shared.lock().await;
                    (s.approval_policy, s.workdir.clone(), s.skills.clone(), s.context.clone(), s.todo.clone(), s.mode)
                };
                registry = Arc::new(ToolRegistry::default_tools_with(policy, extra));

                // Refresh the system prompt (messages[0]) so the model sees the
                // updated tool list, keeping the current skills + context.
                let tool_refs: Vec<(String, String)> = {
                    let owned = registry.definitions();
                    owned
                        .into_iter()
                        .map(|d| (d.function.name, d.function.description))
                        .collect()
                };
                let skill_refs: Vec<(&str, &str, &str)> = skills
                    .iter()
                    .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                    .collect();
                let context_refs: Vec<(&str, &str, &str)> = context
                    .iter()
                    .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                    .collect();
                let refs: Vec<(&str, &str)> = tool_refs
                    .iter()
                    .map(|(a, b)| (a.as_str(), b.as_str()))
                    .collect();
                let ctx = ToolContext {
                    workdir,
                    os: std::env::consts::OS.to_string(),
                };
                let new_prompt =
                    prompt::build_system_prompt(&ctx, &refs, &skill_refs, &context_refs, &todo, mode);
                let mut s = shared.lock().await;
                s.enabled_user_tools = tools;
                if let Some(first) = s.messages.first_mut() {
                    first.content = new_prompt;
                } else {
                    s.messages.insert(0, ChatMessage::system(new_prompt));
                }
                drop(s);
                let _ = event_tx
                    .send(AgentEvent::Status {
                        message: format!("Tools reloaded ({count} user tools enabled)."),
                    })
                    .await;
            }
            Command::ReloadContext { context } => {
                let count = context.len();
                // Rebuild the system prompt with current tools + skills + the new
                // context set. Mirrors ReloadSkills: replace messages[0] in place,
                // preserve the rest of the conversation, update Shared.context.
                let tool_refs: Vec<(String, String)> = {
                    let owned = registry.definitions();
                    owned
                        .into_iter()
                        .map(|d| (d.function.name, d.function.description))
                        .collect()
                };
                let mut s = shared.lock().await;
                s.context = context.clone();
                let skill_refs: Vec<(&str, &str, &str)> = s
                    .skills
                    .iter()
                    .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                    .collect();
                let context_refs: Vec<(&str, &str, &str)> = context
                    .iter()
                    .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
                    .collect();
                let refs: Vec<(&str, &str)> = tool_refs
                    .iter()
                    .map(|(a, b)| (a.as_str(), b.as_str()))
                    .collect();
                let ctx = ToolContext {
                    workdir: s.workdir.clone(),
                    os: std::env::consts::OS.to_string(),
                };
                let new_prompt = prompt::build_system_prompt(
                    &ctx, &refs, &skill_refs, &context_refs, &s.todo, s.mode,
                );
                if let Some(first) = s.messages.first_mut() {
                    first.content = new_prompt;
                } else {
                    s.messages.insert(0, ChatMessage::system(new_prompt));
                }
                drop(s);
                let _ = event_tx
                    .send(AgentEvent::Status {
                        message: format!("Context reloaded ({count} files enabled)."),
                    })
                    .await;
            }
            Command::ReloadMcp { connections } => {
                // Reconnect to every enabled MCP server, collect their tools,
                // and rebuild the registry (built-ins + user scripts + MCP
                // tools). Same between-turns safety as ReloadTools. Connection
                // failures are non-fatal: a bad server is logged + skipped so
                // one broken server can't brick the agent.
                let count = connections.len();
                let (policy, workdir) = {
                    let s = shared.lock().await;
                    (s.approval_policy, s.workdir.clone())
                };
                let extra = connect_mcp_and_user_tools(&connections, &shared, policy).await;

                // Update Shared.mcp, then rebuild the registry + prompt.
                let mut s = shared.lock().await;
                s.mcp = connections;
                registry = Arc::new(ToolRegistry::default_tools_with(policy, extra));
                drop(s);

                rebuild_prompt_from(&shared, &registry, &workdir).await;
                let _ = event_tx
                    .send(AgentEvent::Status {
                        message: format!("MCP reloaded ({count} servers)."),
                    })
                    .await;
            }
        }
    }
}

/// Connect to every MCP server in `specs`, collect their tools, and prepend
/// the active profile's user-script tools — returning the combined `extra` Vec
/// for `ToolRegistry::default_tools_with`. Server failures are logged via the
/// returned count and skipped (their tools are simply absent). Also refreshes
/// `Shared.enabled_user_tools` is NOT done here (that's the caller's job for
/// ReloadTools); this only reads the current user-tool specs from Shared.
async fn connect_mcp_and_user_tools(
    specs: &[McpConnectionSpec],
    shared: &Arc<Mutex<Shared>>,
    policy: ApprovalPolicy,
) -> Vec<Box<dyn super::tools::Tool>> {
    // Start from the profile's user-script tools.
    let user_specs = shared.lock().await.enabled_user_tools.clone();
    let mut extra: Vec<Box<dyn super::tools::Tool>> = build_user_tools(
        &user_specs.iter().flat_map(|s| s.to_rows()).collect::<Vec<_>>(),
    );

    for spec in specs {
        match mcp::McpClient::connect(spec.to_transport()).await {
            Ok(client) => match mcp::build_mcp_tools(client).await {
                Ok((tools, resources)) => {
                    if !resources.is_empty() {
                        tracing::info!(
                            "MCP server '{}' exposes {} resources (v1: tools-only, ignored)",
                            spec.name,
                            resources.len()
                        );
                    }
                    tracing::info!(
                        "MCP server '{}' connected with {} tools",
                        spec.name,
                        tools.len()
                    );
                    extra.extend(tools);
                }
                Err(e) => {
                    tracing::warn!("MCP server '{}' tool discovery failed: {e}", spec.name);
                }
            },
            Err(e) => {
                tracing::warn!("MCP server '{}' failed to connect: {e}", spec.name);
            }
        }
    }
    // policy is accepted to keep the signature stable; the registry uses it.
    let _ = policy;
    extra
}

/// Rebuild `messages[0]` from the current registry + Shared fields. Mirrors the
/// inline logic in ReloadSkills/ReloadTools/ReloadContext so the prompt always
/// reflects the live tool set.
async fn rebuild_prompt_from(
    shared: &Arc<Mutex<Shared>>,
    registry: &Arc<ToolRegistry>,
    workdir: &std::path::Path,
) {
    let (skills, context, todo, mode) = {
        let s = shared.lock().await;
        (s.skills.clone(), s.context.clone(), s.todo.clone(), s.mode)
    };
    let tool_refs: Vec<(String, String)> = registry
        .definitions()
        .into_iter()
        .map(|d| (d.function.name, d.function.description))
        .collect();
    let skill_refs: Vec<(&str, &str, &str)> = skills
        .iter()
        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
        .collect();
    let context_refs: Vec<(&str, &str, &str)> = context
        .iter()
        .map(|(a, b, c)| (a.as_str(), b.as_str(), c.as_str()))
        .collect();
    let refs: Vec<(&str, &str)> = tool_refs
        .iter()
        .map(|(a, b)| (a.as_str(), b.as_str()))
        .collect();
    let ctx = ToolContext {
        workdir: workdir.to_path_buf(),
        os: std::env::consts::OS.to_string(),
    };
    let new_prompt =
        prompt::build_system_prompt(&ctx, &refs, &skill_refs, &context_refs, &todo, mode);
    let mut s = shared.lock().await;
    if let Some(first) = s.messages.first_mut() {
        first.content = new_prompt;
    } else {
        s.messages.insert(0, ChatMessage::system(new_prompt));
    }
}

#[allow(clippy::too_many_arguments)]
async fn handle_user_input(
    provider: &Arc<dyn ModelProvider>,
    registry: &Arc<ToolRegistry>,
    store: &Arc<Mutex<crate::db::MemoryStore>>,
    shared: &Arc<Mutex<Shared>>,
    event_tx: &mpsc::Sender<AgentEvent>,
    text: String,
) -> Result<()> {
    // Snapshot the live settings (model, workdir, behavior) once per turn.
    let (model, workdir, max_iterations, context_window) = {
        let s = shared.lock().await;
        (
            s.model.clone(),
            s.workdir.clone(),
            s.max_iterations,
            s.context_window,
        )
    };
    let ctx = ToolContext {
        workdir: workdir.clone(),
        os: std::env::consts::OS.to_string(),
    };

    // Ensure a session exists for this project.
    {
        let mut s = shared.lock().await;
        if s.session_id.is_none() {
            let project_id = store
                .lock()
                .await
                .ensure_project(&workdir.to_string_lossy(), None)
                .ok();
            let title = derive_title(&text);
            let sid = store
                .lock()
                .await
                .start_session(project_id, &model, &title)?;
            s.session_id = Some(sid);
        }
    }

    // Append + persist the user message.
    let user_msg = ChatMessage::user(&text);
    {
        let mut s = shared.lock().await;
        persist_message(store, &mut s, &user_msg, None, None, None).await;
        s.messages.push(user_msg);
    }

    // Run the reasoning loop.
    let mut iterations = 0u32;
    let mode = shared.lock().await.mode;
    let max = max_iterations;

    loop {
        if iterations >= max {
            let _ = event_tx
                .send(AgentEvent::Status {
                    message: format!("Reached iteration cap ({max}); stopping."),
                })
                .await;
            break;
        }
        iterations += 1;

        let request = ChatRequest {
            messages: {
                let s = shared.lock().await;
                trim_to_context(s.messages.clone(), context_window)
            },
            tools: tools_with_subagents(&registry, &store).await,
            temperature: 0.2,
        };

        // Stream the model turn.
        let mut rx = provider.chat(&model, request).await?;
        let mut text = String::new();
        // Model reasoning/"thinking", surfaced separately as a collapsible block.
        let mut reasoning = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut usage_in: Option<i64> = None;
        let mut usage_out: Option<i64> = None;

        while let Some(ev) = rx.recv().await {
            match ev {
                ChatEvent::Reasoning(r) => {
                    reasoning.push_str(&r);
                    let _ = event_tx.send(AgentEvent::AssistantReasoning { delta: r }).await;
                }
                ChatEvent::Delta(d) => {
                    text.push_str(&d);
                    let _ = event_tx.send(AgentEvent::AssistantDelta { delta: d }).await;
                }
                ChatEvent::ToolCalls(tc) => {
                    tool_calls = tc;
                }
                ChatEvent::Done(u) => {
                    usage_in = Some(u.input_tokens);
                    usage_out = Some(u.output_tokens);
                }
                ChatEvent::Error(e) => {
                    let _ = event_tx.send(AgentEvent::Error { message: e }).await;
                    let _ = event_tx.send(AgentEvent::TurnDone { iterations }).await;
                    return Ok(());
                }
            }
        }

        // Build and persist the assistant message.
        let assistant_msg = ChatMessage {
            role: ChatRole::Assistant,
            content: text.clone(),
            tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls.clone()) },
            tool_name: None,
        };
        {
            let mut s = shared.lock().await;
            persist_message(
                store,
                &mut s,
                &assistant_msg,
                Some(&model),
                usage_in,
                usage_out,
            )
            .await;
            s.messages.push(assistant_msg);
        }
        let reasoning_final = if reasoning.is_empty() { None } else { Some(reasoning) };
        let _ = event_tx
            .send(AgentEvent::AssistantMessage {
                text: text.clone(),
                tool_calls: tool_calls.clone(),
                reasoning: reasoning_final,
            })
            .await;

        // No tool calls => final answer.
        if tool_calls.is_empty() {
            let _ = event_tx.send(AgentEvent::TurnDone { iterations }).await;
            return Ok(());
        }

        // Auto-mode cycling valve: if we've done >4 tool rounds and are still
        // calling tools (not finishing), we're likely stuck — pause and ask the
        // user to clarify or confirm we should continue.
        if mode == crate::config::Mode::Auto && iterations > 4 {
            let _ = event_tx
                .send(AgentEvent::Status {
                    message: format!(
                        "Auto mode: paused after {iterations} tool steps — I may be cycling. \
                         Please clarify or tell me to continue."
                    ),
                })
                .await;
            let _ = event_tx.send(AgentEvent::TurnDone { iterations }).await;
            return Ok(());
        }

        // Execute each tool call, honoring the approval gate.
        for (index, call) in tool_calls.iter().enumerate() {
            let name = call.function.name.clone();
            let args = parse_arguments(&call.function.arguments).unwrap_or(serde_json::Value::Null);
            let args_json = serde_json::to_string(&args).unwrap_or_default();
            let started = std::time::Instant::now();

            let _ = event_tx
                .send(AgentEvent::ToolStarted {
                    index,
                    name: name.clone(),
                    args: args_json.clone(),
                })
                .await;

            // Approval gate for write tools.
            let approved = if registry.requires_approval(&name)? {
                let (tx, rx) = oneshot::channel();
                {
                    let mut s = shared.lock().await;
                    s.pending = Some(PendingApproval {
                        index,
                        call: call.clone(),
                        responder: tx,
                    });
                }
                let _ = event_tx
                    .send(AgentEvent::ToolNeedsApproval {
                        index,
                        name: name.clone(),
                        args: args_json.clone(),
                    })
                    .await;
                rx.await.unwrap_or(false)
            } else {
                true
            };

            let result = if approved {
                if name == "delegate" {
                    run_delegation(&provider, &store, &model, &args, index, event_tx).await
                } else if name == "update_todo" {
                    run_update_todo(&store, &shared, &registry, &args, event_tx).await
                } else {
                    registry.execute(&name, &args, &ctx).await
                }
            } else {
                Ok(crate::agent::tools::ToolResult::err(
                    "Denied by user.".to_string(),
                ))
            };

            let outcome = match result {
                Ok(r) => r,
                Err(e) => crate::agent::tools::ToolResult::err(e.to_string()),
            };

            let _ = event_tx
                .send(AgentEvent::ToolFinished {
                    index,
                    name: name.clone(),
                    success: outcome.success,
                    result: outcome.content.clone(),
                    duration_ms: Some(started.elapsed().as_millis() as u64),
                })
                .await;

            // Persist tool result and append to conversation.
            let tool_msg = ChatMessage::tool_result(&name, &outcome.content);
            {
                let mut s = shared.lock().await;
                persist_message(store, &mut s, &tool_msg, None, None, None).await;
                s.messages.push(tool_msg);
            }
        }

        // Loop again: the model will see the tool results.
    }

    let _ = event_tx.send(AgentEvent::TurnDone { iterations }).await;
    Ok(())
}

/// Execute an `update_todo` tool call: replace the to-do list markdown,
/// persist it to the DB, rebuild the system prompt, and notify the UI. The
/// to-do list is shared ground truth — the model maintains it while working,
/// and the user may edit it directly in Plan mode.
async fn run_update_todo(
    store: &Arc<Mutex<crate::db::MemoryStore>>,
    shared: &Arc<Mutex<Shared>>,
    registry: &Arc<ToolRegistry>,
    args: &serde_json::Value,
    event_tx: &mpsc::Sender<AgentEvent>,
) -> Result<crate::agent::tools::ToolResult> {
    let markdown = args
        .get("markdown")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            crate::error::PhoenixError::Other(
                "update_todo: missing 'markdown' argument".into(),
            )
        })?
        .to_string();
    if markdown.trim().is_empty() {
        return Ok(crate::agent::tools::ToolResult::err(
            "The 'markdown' argument is empty — send the full updated to-do list.",
        ));
    }
    // Persist so the plan survives restarts, then update Shared + the prompt.
    {
        let s = store.lock().await;
        if let Err(e) = s.set_setting("todo_markdown", &markdown) {
            tracing::warn!("persisting the to-do list failed: {e}");
        }
    }
    let workdir = {
        let mut s = shared.lock().await;
        s.todo = markdown.clone();
        s.workdir.clone()
    };
    rebuild_prompt_from(shared, registry, &workdir).await;
    let _ = event_tx
        .send(AgentEvent::TodoUpdated {
            markdown: markdown.clone(),
        })
        .await;
    Ok(crate::agent::tools::ToolResult::ok(
        "To-do list updated — the user now sees this plan in the to-do panel.",
    ))
}

/// Build the tool list for a model turn, rewriting the `delegate` tool's
/// description to name the currently-defined sub-agents (read live from the
/// store) so the model knows who it can call.
async fn tools_with_subagents(
    registry: &Arc<ToolRegistry>,
    store: &Arc<Mutex<crate::db::MemoryStore>>,
) -> Vec<crate::model::ToolDef> {
    let mut defs = registry.definitions();
    let agents = store.lock().await.list_sub_agents().unwrap_or_default();
    let listing = if agents.is_empty() {
        "none defined yet (create some in the Sub-Agents tab)".to_string()
    } else {
        agents
            .iter()
            .map(|a| {
                format!(
                    "{} — {}",
                    a.name,
                    if a.description.trim().is_empty() {
                        "(specialist)"
                    } else {
                        &a.description
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    for d in defs.iter_mut() {
        if d.function.name == "delegate" {
            d.function.description = format!(
                "Delegate a specialist sub-task to a predefined sub-agent. Call with \
                 {{\"sub_agent\":\"<name>\",\"task\":\"<...>\"}}. Available sub-agents: \
                 {listing}. Use the exact name."
            );
        }
    }
    defs
}

/// Execute a `delegate` tool call: look up the named sub-agent, run it (its
/// persona + model) on a one-shot nested chat, and return its answer as the tool
/// result. The sub-agent's model defaults to the active model when blank.
///
/// The nested chat's streamed output (text + reasoning) is forwarded to the UI
/// as `SubAgentDelta` / `SubAgentReasoning` so the user can watch the sub-agent
/// work, nested inside the `delegate` tool card. `index` is the tool-call index
/// used to correlate these events with the parent `ToolStarted`/`ToolFinished`.
async fn run_delegation(
    provider: &Arc<dyn ModelProvider>,
    store: &Arc<Mutex<crate::db::MemoryStore>>,
    active_model: &str,
    args: &serde_json::Value,
    index: usize,
    event_tx: &mpsc::Sender<AgentEvent>,
) -> Result<crate::agent::tools::ToolResult> {
    let sub_name = args
        .get("sub_agent")
        .and_then(|v| v.as_str())
        .ok_or_else(|| crate::error::PhoenixError::Other("delegate: missing 'sub_agent'".into()))?;
    let task = args
        .get("task")
        .and_then(|v| v.as_str())
        .ok_or_else(|| crate::error::PhoenixError::Other("delegate: missing 'task'".into()))?;

    let agents = store.lock().await.list_sub_agents()?;
    let Some(sa) = agents
        .iter()
        .find(|a| a.name.eq_ignore_ascii_case(sub_name))
        .cloned()
    else {
        let names: Vec<_> = agents.iter().map(|a| a.name.clone()).collect();
        return Ok(crate::agent::tools::ToolResult::err(format!(
            "No sub-agent named '{sub_name}'. Available: {}",
            names.join(", ")
        )));
    };

    let persona = if sa.persona.trim().is_empty() {
        format!("You are {}, a specialist sub-agent. {}", sa.name, sa.description)
    } else {
        sa.persona.clone()
    };
    let model = if sa.model.trim().is_empty() {
        active_model.to_string()
    } else {
        sa.model.clone()
    };

    // Tell the UI a sub-agent is starting (rendered nested in the tool card).
    let _ = event_tx
        .send(AgentEvent::SubAgentStarted {
            index,
            name: sa.name.clone(),
            model: model.clone(),
            task: task.to_string(),
        })
        .await;

    let request = crate::model::ChatRequest {
        messages: vec![
            crate::model::ChatMessage::system(format!(
                "{persona}\n\nYou are answering a delegated sub-task for the main agent. \
                 Respond with a focused, self-contained answer — do not ask questions."
            )),
            crate::model::ChatMessage::user(format!("Task: {task}")),
        ],
        tools: Vec::new(),
        temperature: 0.2,
    };
    let mut rx = provider.chat(&model, request).await?;
    let mut out = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            crate::model::ChatEvent::Delta(d) => {
                out.push_str(&d);
                let _ = event_tx
                    .send(AgentEvent::SubAgentDelta {
                        index,
                        name: sa.name.clone(),
                        text: d,
                    })
                    .await;
            }
            crate::model::ChatEvent::Reasoning(r) => {
                let _ = event_tx
                    .send(AgentEvent::SubAgentReasoning {
                        index,
                        name: sa.name.clone(),
                        text: r,
                    })
                    .await;
            }
            _ => {}
        }
    }

    let result_text = if out.trim().is_empty() {
        let msg = "Sub-agent returned an empty response.".to_string();
        let _ = event_tx
            .send(AgentEvent::SubAgentFinished {
                index,
                name: sa.name.clone(),
                model: model.clone(),
                result: msg.clone(),
            })
            .await;
        return Ok(crate::agent::tools::ToolResult::err(msg));
    } else {
        format!("[{} · {}]: {out}", sa.name, model)
    };

    let _ = event_tx
        .send(AgentEvent::SubAgentFinished {
            index,
            name: sa.name.clone(),
            model: model.clone(),
            result: result_text.clone(),
        })
        .await;
    Ok(crate::agent::tools::ToolResult::ok(result_text))
}

/// Persist a message to the active session (no-op if no session yet).
async fn persist_message(
    store: &Arc<Mutex<crate::db::MemoryStore>>,
    shared: &mut Shared,
    msg: &ChatMessage,
    model: Option<&str>,
    tokens_in: Option<i64>,
    tokens_out: Option<i64>,
) {
    if let Some(sid) = shared.session_id {
        let store = store.lock().await;
        let _ = store.append_message(sid, msg, model, tokens_in, tokens_out);
    }
}

/// Keep the system prompt + the last `window` non-system messages.
fn trim_to_context(mut messages: Vec<ChatMessage>, window: u32) -> Vec<ChatMessage> {
    if messages.len() <= 1 {
        return messages;
    }
    let sys = messages.remove(0);
    let start = messages.len().saturating_sub(window as usize);
    let mut out = vec![sys];
    out.extend(messages.into_iter().skip(start));
    out
}

/// Derive a short session title from the first user message.
fn derive_title(text: &str) -> String {
    let t = text.trim().replace('\n', " ");
    let head: String = t.chars().take(60).collect();
    if head.chars().count() < t.chars().count() {
        format!("{head}…")
    } else {
        head
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All `AgentEvent` variants must serialize as `{"type": "<snake>", ...named}`
    /// so the frontend can read payloads by field name (no serde `"0"` quirk).
    #[test]
    fn agent_event_serializes_named_fields() {
        let v = serde_json::to_value(AgentEvent::AssistantDelta { delta: "hi".into() }).unwrap();
        assert_eq!(v["type"], "assistant_delta");
        assert_eq!(v["delta"], "hi");

        let v = serde_json::to_value(AgentEvent::AssistantReasoning { delta: "r".into() }).unwrap();
        assert_eq!(v["type"], "assistant_reasoning");
        assert_eq!(v["delta"], "r");

        let v = serde_json::to_value(AgentEvent::Status { message: "ok".into() }).unwrap();
        assert_eq!(v["type"], "status");
        assert_eq!(v["message"], "ok");
        // The converted newtype variants must NOT leak the old numeric `"0"` key.
        assert!(v.get("0").is_none());

        let v = serde_json::to_value(AgentEvent::Error { message: "boom".into() }).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["message"], "boom");

        let v = serde_json::to_value(AgentEvent::ToolFinished {
            index: 2,
            name: "read_file".into(),
            success: true,
            result: "ok".into(),
            duration_ms: Some(42),
        })
        .unwrap();
        assert_eq!(v["type"], "tool_finished");
        assert_eq!(v["index"], 2);
        assert_eq!(v["duration_ms"], 42);

        let v = serde_json::to_value(AgentEvent::SubAgentStarted {
            index: 0,
            name: "Researcher".into(),
            model: "qwen3:8b".into(),
            task: "find x".into(),
        })
        .unwrap();
        assert_eq!(v["type"], "sub_agent_started");
        assert_eq!(v["model"], "qwen3:8b");
    }
}

