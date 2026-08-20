//! User-installed executable tools (Panel 3).
//!
//! A `UserScriptTool` wraps a script the user authored or installed. When the
//! model calls it, the runtime:
//!   1. writes the script body to a temp file with a language-appropriate
//!      extension,
//!   2. spawns the configured interpreter with that file as its first argument,
//!   3. feeds the model's arguments as JSON on stdin,
//!   4. captures stdout (+ stderr), applies a timeout, and returns the text.
//!
//! ## Security
//! User scripts run with the *user's* full privileges — exactly like the
//! built-in `run_command` tool. The guardrails are: (a) the approval gate
//! (`WritesOnly`/`All` policy — write-kind tools prompt before running), and
//! (b) the user explicitly installed/enabled the tool. This adds no privilege
//! the agent didn't already have via `run_command`; it just makes a custom
//! script callable by name with a declared schema. See the Panel 3 plan's
//! "out of scope" note for sandboxing (future work).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::AsyncWriteExt;

use crate::config::ToolKind;

use super::{Tool, ToolContext, ToolResult};

/// Maximum output size (chars) returned to the model before truncation.
const MAX_OUTPUT: usize = 20_000;
/// Wall-clock timeout for a user script, in seconds.
const TIMEOUT_SECS: u64 = 60;

/// A user-provided script registered as a callable tool.
pub struct UserScriptTool {
    name: String,
    description: String,
    interpreter: String,
    script_body: Arc<str>,
    params_schema: Value,
    kind: ToolKind,
}

impl UserScriptTool {
    pub fn new(
        name: &str,
        description: &str,
        interpreter: &str,
        script_body: &str,
        params_schema: Value,
        kind: ToolKind,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            interpreter: interpreter.to_string(),
            script_body: Arc::from(script_body),
            params_schema,
            kind,
        }
    }
}

#[async_trait]
impl Tool for UserScriptTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Value {
        self.params_schema.clone()
    }

    fn kind(&self) -> ToolKind {
        self.kind
    }

    async fn run(&self, args: &Value, _ctx: &ToolContext) -> ToolResult {
        // 1. Write the script to a temp file with a sensible extension so the
        //    interpreter is happy (e.g. python needs `.py` for some imports).
        let ext = extension_for(&self.interpreter);
        let tmp = match tempfile::Builder::new().suffix(&ext).tempfile() {
            Ok(f) => f,
            Err(e) => return ToolResult::err(format!("create temp script file: {e}")),
        };
        // Keep the file on disk (we delete it ourselves below) so we can spawn
        // the interpreter against a real path.
        let path = match tmp.keep() {
            Ok((_file, path)) => {
                drop(_file);
                path
            }
            Err(e) => return ToolResult::err(format!("persist temp script file: {e}")),
        };

        if let Err(e) = std::fs::write(&path, self.script_body.as_ref()) {
            let _ = std::fs::remove_file(&path);
            return ToolResult::err(format!("write temp script file: {e}"));
        }

        // 2. Build the interpreter command. Known names get the right argv
        //    shape; anything else is treated as a program taking the file as
        //    its first argument.
        let (program, argv) = command_for(&self.interpreter, &path);

        // Cleanup helper: best-effort remove the temp file in every path.
        let cleanup = || {
            let _ = std::fs::remove_file(&path);
        };

        // 3. Spawn with piped stdio.
        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&argv);
        cmd.stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                cleanup();
                return ToolResult::err(format!(
                    "spawn interpreter '{program}' (is '{}' installed and on PATH?): {e}",
                    self.interpreter
                ));
            }
        };

        // 4. Write the args JSON to stdin, then close it.
        let payload = serde_json::to_string_pretty(args).unwrap_or_else(|_| "{}".into());
        if let Some(mut stdin) = child.stdin.take() {
            // Spawn the write so the child can't deadlock waiting on a full pipe.
            let p = payload.clone();
            tokio::spawn(async move {
                let _ = stdin.write_all(p.as_bytes()).await;
                let _ = stdin.shutdown().await;
            });
        }

        // 5. Wait with a timeout.
        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(TIMEOUT_SECS),
            child.wait_with_output(),
        )
        .await;

        match outcome {
            Ok(Ok(out)) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                let code = out.status.code().unwrap_or(-1);
                let mut combined = String::new();
                if !stdout.is_empty() {
                    combined.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !combined.is_empty() {
                        combined.push_str("\n--- stderr ---\n");
                    }
                    combined.push_str(&stderr);
                }
                if combined.trim().is_empty() {
                    combined = format!("(no output; exited with code {code})");
                }
                let combined = trim_to(&combined, MAX_OUTPUT);
                cleanup();
                ToolResult {
                    success: out.status.success(),
                    content: format!(
                        "tool `{}` exited with code {code}\n{combined}",
                        self.name
                    ),
                }
            }
            Ok(Err(e)) => {
                cleanup();
                ToolResult::err(format!("wait: {e}"))
            }
            Err(_) => {
                cleanup();
                ToolResult::err(format!(
                    "tool `{}` timed out after {TIMEOUT_SECS}s and was killed",
                    self.name
                ))
            }
        }
    }
}

/// Pick a temp-file suffix for the given interpreter name.
fn extension_for(interpreter: &str) -> String {
    match interpreter.trim() {
        "python" | "python3" | "py" => ".py".into(),
        "node" | "nodejs" => ".js".into(),
        "sh" | "bash" | "zsh" => ".sh".into(),
        "powershell" | "pwsh" => ".ps1".into(),
        _ => ".txt".into(),
    }
}

/// Build the (program, argv) tuple for the interpreter + script path.
fn command_for(interpreter: &str, script_path: &std::path::Path) -> (String, Vec<String>) {
    let p = script_path.to_string_lossy().to_string();
    match interpreter.trim() {
        "python" | "python3" | "py" => ("python".into(), vec![p]),
        "node" | "nodejs" => ("node".into(), vec![p]),
        "sh" => ("sh".into(), vec![p]),
        "bash" => ("bash".into(), vec![p]),
        "zsh" => ("zsh".into(), vec![p]),
        "powershell" | "pwsh" => ("powershell".into(), vec!["-File".into(), p]),
        other => (other.to_string(), vec![p]),
    }
}

/// Truncate a string to `max` chars with an ellipsis indicator.
fn trim_to(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}\n…[output truncated, {} chars total]", &s[..max], s.len())
    }
}
