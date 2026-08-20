//! Model Context Protocol (MCP) client — Panel 5 ("memory").
//!
//! An MCP server is an external process (stdio) or HTTP endpoint that exposes
//! **tools** (and optionally resources/prompts) the agent can call. This module
//! implements just enough of the MCP JSON-RPC 2.0 protocol to:
//!
//! 1. `initialize` a connection (handshake + `notifications/initialized`),
//! 2. `tools/list` to discover the server's tools, and
//! 3. `tools/call` to invoke one.
//!
//! Each discovered tool is wrapped in an [`McpTool`] that implements the agent's
//! [`Tool`](crate::agent::tools::Tool) trait, so MCP tools register alongside
//! the built-ins and user-script tools in the [`ToolRegistry`].
//!
//! Design notes:
//! - Errors are wrapped in `PhoenixError::Other(format!(...))` (matching
//!   `skills.rs`) so failures are human-readable. A broken connection surfaces
//!   to the model as a failed tool result rather than panicking the runtime.
//! - The stdio transport uses a request-id counter and a pending-request map so
//!   that (future) notifications from the server don't desync responses. For v1
//!   we only issue one request at a time per client, which keeps framing simple.
//!
//! [`ToolRegistry`]: crate::agent::tools::ToolRegistry

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin};
use tokio::sync::{Mutex, oneshot};

use crate::agent::tools::{Tool, ToolContext, ToolResult};
use crate::config::ToolKind;
use crate::error::{PhoenixError, Result};

/// How an MCP server is reached.
#[derive(Debug, Clone)]
pub enum McpTransport {
    /// Spawn a child process and speak JSON-RPC 2.0 over its stdin/stdout
    /// (newline-delimited UTF-8 JSON, one message per line).
    Stdio { command: String, args: Vec<String> },
    /// POST JSON-RPC 2.0 requests to a base URL.
    Http { url: String },
}

/// One tool advertised by an MCP server (`tools/list` result).
#[derive(Debug, Clone)]
pub struct McpToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema describing the tool's parameters object.
    pub input_schema: Value,
}

/// A live connection to an MCP server. Cheap to clone via the inner `Arc`.
///
/// For stdio, the child process and its stdin are owned here; the stdout reader
/// task routes responses back to callers via a shared pending-request table.
#[derive(Clone)]
pub struct McpClient {
    inner: Arc<ClientInner>,
}

enum ClientInner {
    /// Long-lived stdio child with a response-router task.
    Stdio {
        stdin: Arc<Mutex<ChildStdin>>,
        next_id: AtomicU64,
        pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>,
        // Held to keep the stdout reader task alive; not directly read after spawn.
        _child: Arc<Mutex<Child>>,
    },
    /// Stateless HTTP transport.
    Http {
        client: reqwest::Client,
        url: String,
        next_id: AtomicU64,
    },
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &*self.inner {
            ClientInner::Stdio { .. } => f.debug_struct("McpClient").field("transport", &"stdio").finish(),
            ClientInner::Http { url, .. } => f
                .debug_struct("McpClient")
                .field("transport", &"http")
                .field("url", url)
                .finish(),
        }
    }
}

impl McpClient {
    /// Establish a connection and complete the MCP `initialize` handshake.
    ///
    /// For stdio this spawns the child and a stdout-reader task that routes
    /// JSON-RPC responses to their awaiting callers. For HTTP it stores the
    /// client lazily (the handshake happens as part of `connect`).
    pub async fn connect(transport: McpTransport) -> Result<Self> {
        let inner = match transport.clone() {
            McpTransport::Stdio { command, args } => {
                let mut child = tokio::process::Command::new(&command)
                    .args(&args)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                    .map_err(|e| {
                        PhoenixError::Other(format!("MCP stdio spawn '{command}': {e}"))
                    })?;
                let stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| PhoenixError::Other("MCP stdio: no stdin".into()))?;
                let stdout = child
                    .stdout
                    .take()
                    .ok_or_else(|| PhoenixError::Other("MCP stdio: no stdout".into()))?;

                let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>> =
                    Arc::new(Mutex::new(HashMap::new()));
                let pending_for_reader = pending.clone();
                tokio::spawn(async move {
                    let mut reader = BufReader::new(stdout).lines();
                    while let Ok(Some(line)) = reader.next_line().await {
                        if line.trim().is_empty() {
                            continue;
                        }
                        // Parse id, then deliver to whoever is waiting (if anyone).
                        if let Ok(v) = serde_json::from_str::<Value>(&line) {
                            if let Some(id) = v.get("id").and_then(|i| i.as_u64()) {
                                let mut map = pending_for_reader.lock().await;
                                if let Some(tx) = map.remove(&id) {
                                    let _ = tx.send(v);
                                }
                            }
                        }
                        // Notifications (no id) are ignored for v1.
                    }
                });

                ClientInner::Stdio {
                    stdin: Arc::new(Mutex::new(stdin)),
                    next_id: AtomicU64::new(1),
                    pending,
                    _child: Arc::new(Mutex::new(child)),
                }
            }
            McpTransport::Http { url } => {
                let client = reqwest::Client::builder()
                    .build()
                    .map_err(|e| PhoenixError::Other(format!("MCP http client build: {e}")))?;
                ClientInner::Http {
                    client,
                    url,
                    next_id: AtomicU64::new(1),
                }
            }
        };

        let client = Self { inner: Arc::new(inner) };

        // Initialize handshake: declare our capabilities, expect the server's.
        let init_result = client
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": { "name": "phoenix-agent", "version": env!("CARGO_PKG_VERSION") },
                }),
            )
            .await
            .map_err(|e| PhoenixError::Other(format!("MCP initialize: {e}")))?;

        // Notify the server we're ready (notification = no id, no response).
        let _ = client.notify("notifications/initialized", json!({})).await;

        // Best-effort: log server info if present (aids debugging).
        if let Some(info) = init_result.get("serverInfo") {
            let name = info.get("name").and_then(|n| n.as_str()).unwrap_or("?");
            let ver = info.get("version").and_then(|v| v.as_str()).unwrap_or("?");
            tracing::info!("MCP server connected: {name} v{ver}");
        }

        Ok(client)
    }

    /// Send a JSON-RPC 2.0 request and await its result field.
    async fn request(&self, method: &str, params: Value) -> Result<Value> {
        let id = match &*self.inner {
            ClientInner::Stdio { next_id, .. } => next_id.fetch_add(1, Ordering::Relaxed),
            ClientInner::Http { next_id, .. } => next_id.fetch_add(1, Ordering::Relaxed),
        };
        let req = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });

        match &*self.inner {
            ClientInner::Stdio { stdin, pending, .. } => {
                let (tx, rx) = oneshot::channel();
                {
                    let mut map = pending.lock().await;
                    map.insert(id, tx);
                }
                let mut line = serde_json::to_string(&req)
                    .map_err(|e| PhoenixError::Other(format!("MCP encode: {e}")))?;
                line.push('\n');
                {
                    let mut s = stdin.lock().await;
                    s.write_all(line.as_bytes())
                        .await
                        .map_err(|e| PhoenixError::Other(format!("MCP stdio write: {e}")))?;
                    s.flush()
                        .await
                        .map_err(|e| PhoenixError::Other(format!("MCP stdio flush: {e}")))?;
                }
                // Await the response, with cleanup on either timeout or error so
                // no waiter is ever left dangling in the pending map.
                let resp = tokio::time::timeout(std::time::Duration::from_secs(30), rx).await;
                let resp = match resp {
                    Ok(Ok(v)) => v,
                    Ok(Err(_)) => {
                        pending.lock().await.remove(&id);
                        return Err(PhoenixError::Other("MCP stdio: reader dropped".into()));
                    }
                    Err(_) => {
                        pending.lock().await.remove(&id);
                        return Err(PhoenixError::Other(format!(
                            "MCP stdio: timed out waiting for '{method}'"
                        )));
                    }
                };
                Self::unwrap_result(resp, method)
            }
            ClientInner::Http { client, url, .. } => {
                let resp = client
                    .post(url.as_str())
                    .json(&req)
                    .send()
                    .await
                    .map_err(|e| PhoenixError::Other(format!("MCP http '{method}' send: {e}")))?;
                let status = resp.status();
                let body: Value = resp
                    .json()
                    .await
                    .map_err(|e| PhoenixError::Other(format!("MCP http '{method}' parse (status {status}): {e}")))?;
                Self::unwrap_result(body, method)
            }
        }
    }

    /// Send a JSON-RPC 2.0 notification (no id, no response expected).
    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let req = json!({ "jsonrpc": "2.0", "method": method, "params": params });
        match &*self.inner {
            ClientInner::Stdio { stdin, .. } => {
                let mut line = serde_json::to_string(&req)
                    .map_err(|e| PhoenixError::Other(format!("MCP encode notify: {e}")))?;
                line.push('\n');
                let mut s = stdin.lock().await;
                s.write_all(line.as_bytes())
                    .await
                    .map_err(|e| PhoenixError::Other(format!("MCP stdio notify write: {e}")))?;
                s.flush()
                    .await
                    .map_err(|e| PhoenixError::Other(format!("MCP stdio notify flush: {e}")))?;
            }
            ClientInner::Http { client, url, .. } => {
                let _ = client
                    .post(url.as_str())
                    .json(&req)
                    .send()
                    .await
                    .map_err(|e| PhoenixError::Other(format!("MCP http notify: {e}")))?;
            }
        }
        Ok(())
    }

    /// Pull the `result` out of a JSON-RPC response, or surface `error`.
    fn unwrap_result(resp: Value, method: &str) -> Result<Value> {
        if let Some(err) = resp.get("error") {
            let code = err.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
            let msg = err.get("message").and_then(|m| m.as_str()).unwrap_or("?");
            return Err(PhoenixError::Other(format!(
                "MCP '{method}' error ({code}): {msg}"
            )));
        }
        resp.get("result")
            .cloned()
            .ok_or_else(|| PhoenixError::Other(format!("MCP '{method}': no result field")))
    }

    /// List the tools advertised by this server.
    pub async fn list_tools(&self) -> Result<Vec<McpToolDef>> {
        let result = self.request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| PhoenixError::Other("MCP tools/list: no 'tools' array".into()))?;
        let mut out = Vec::new();
        for t in tools {
            let name = t
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| PhoenixError::Other("MCP tool: missing 'name'".into()))?
                .to_string();
            let description = t
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("")
                .to_string();
            let input_schema = t
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| Value::Object(Default::default()));
            out.push(McpToolDef { name, description, input_schema });
        }
        Ok(out)
    }

    /// Invoke a tool by name. Returns the flattened text content.
    pub async fn call_tool(&self, name: &str, args: &Value) -> Result<String> {
        let result = self
            .request("tools/call", json!({ "name": name, "arguments": args }))
            .await?;
        // MCP returns `{ "content": [ { "type": "text", "text": "..." }, ... ], "isError"?: bool }`.
        let is_error = result
            .get("isError")
            .and_then(|e| e.as_bool())
            .unwrap_or(false);
        let content = result
            .get("content")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        let mut parts: Vec<String> = Vec::new();
        for item in content {
            if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                    parts.push(text.to_string());
                }
            } else {
                // Non-text content (image/resource embeds) — include a stub so the
                // model at least knows something was returned.
                parts.push(format!("[non-text content: {item}]"));
            }
        }
        let joined = parts.join("\n");
        if is_error {
            Err(PhoenixError::Other(format!(
                "MCP tool '{name}' reported error: {joined}"
            )))
        } else {
            Ok(joined)
        }
    }

    /// Best-effort discovery of resources. Returns empty if unsupported.
    pub async fn list_resources(&self) -> Vec<String> {
        match self.request("resources/list", json!({})).await {
            Ok(result) => result
                .get("resources")
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|r| r.get("uri").and_then(|u| u.as_str()).map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            Err(_) => Vec::new(), // method-not-found or similar — not fatal.
        }
    }
}

/// A single MCP tool wrapped to satisfy the agent's [`Tool`] trait.
pub struct McpTool {
    def: McpToolDef,
    client: McpClient,
}

impl McpTool {
    pub fn new(def: McpToolDef, client: McpClient) -> Self {
        Self { def, client }
    }
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn name(&self) -> &str {
        &self.def.name
    }

    fn description(&self) -> &str {
        if self.def.description.is_empty() {
            "MCP server tool"
        } else {
            &self.def.description
        }
    }

    fn parameters_schema(&self) -> Value {
        self.def.input_schema.clone()
    }

    fn kind(&self) -> ToolKind {
        // v1: MCP tools are conservatively treated as read-only so they run
        // under the default WritesOnly policy without per-call approval. We do
        // not yet trust server-declared annotations.
        ToolKind::Read
    }

    async fn run(&self, args: &Value, _ctx: &ToolContext) -> ToolResult {
        match self.client.call_tool(&self.def.name, args).await {
            Ok(text) => ToolResult::ok(text),
            Err(e) => ToolResult::err(e.to_string()),
        }
    }
}

/// Connect to a server and build one [`McpTool`] per advertised tool.
///
/// Returns the tools plus the discovered resource URIs (logged by the caller).
/// Used by the runtime at reload time.
pub async fn build_mcp_tools(client: McpClient) -> Result<(Vec<Box<dyn Tool>>, Vec<String>)> {
    let defs = client.list_tools().await?;
    let resources = client.list_resources().await;
    let tools: Vec<Box<dyn Tool>> = defs
        .into_iter()
        .map(|d| Box::new(McpTool::new(d, client.clone())) as Box<dyn Tool>)
        .collect();
    Ok((tools, resources))
}
