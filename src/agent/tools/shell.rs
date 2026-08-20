//! Shell tool: `run_command`.

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{arg_str, arg_u32, Tool, ToolContext, ToolResult};
use crate::config::ToolKind;

/// Run a shell command and return combined stdout/stderr. Mutating — approval-gated.
pub struct RunCommand;
#[async_trait]
impl Tool for RunCommand {
    fn name(&self) -> &str { "run_command" }
    fn description(&self) -> &str {
        "Run a shell command in the project working directory and return stdout+stderr. Use this for builds, tests, git, and other dev tasks. Commands run non-interactively."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The shell command line to run." },
                "timeout_secs": { "type": "integer", "default": 120, "description": "Maximum wall-clock seconds before the command is killed." }
            },
            "required": ["command"]
        })
    }
    fn kind(&self) -> ToolKind { ToolKind::Write }
    async fn run(&self, args: &Value, ctx: &ToolContext) -> ToolResult {
        let Some(command) = arg_str(args, "command") else {
            return ToolResult::err("missing required parameter 'command'");
        };
        let timeout_secs = arg_u32(args, "timeout_secs", 120).max(1) as u64;

        // Use cmd on Windows, sh elsewhere. We deliberately do not pass the
        // command through a login shell: the workdir and explicit cmd are enough.
        let (program, flag) = if cfg!(target_os = "windows") {
            ("cmd", "/C")
        } else {
            ("sh", "-c")
        };

        let mut cmd = tokio::process::Command::new(program);
        cmd.arg(flag).arg(&command).current_dir(&ctx.workdir);
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null());

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return ToolResult::err(format!("spawn {program}: {e}")),
        };

        let outcome = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
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
                let combined = trim_to(&combined, 20_000);
                ToolResult {
                    success: out.status.success(),
                    content: format!(
                        "Command `{command}` exited with code {code}\n{combined}"
                    ),
                }
            }
            Ok(Err(e)) => ToolResult::err(format!("wait: {e}")),
            Err(_) => ToolResult::err(format!(
                "command timed out after {timeout_secs}s and was killed"
            )),
        }
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
