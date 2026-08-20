//! Search tools: `grep` (wraps ripgrep).

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{arg_bool, arg_str, arg_u32, Tool, ToolContext, ToolResult};
use crate::config::ToolKind;

/// Grep for a pattern (literal or regex) under the working directory.
pub struct Grep;
#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str { "grep" }
    fn description(&self) -> &str {
        "Search file contents for a pattern (regular expression by default). Returns matching lines with file:line prefixes. Searches the working directory recursively."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern":    { "type": "string", "description": "Regular expression to search for." },
                "path":       { "type": "string", "description": "File or directory to search; defaults to the working directory." },
                "case_insensitive": { "type": "boolean", "default": false },
                "literal":    { "type": "boolean", "default": false, "description": "Treat pattern as a literal string." },
                "max_matches":{ "type": "integer", "default": 100 }
            },
            "required": ["pattern"]
        })
    }
    fn kind(&self) -> ToolKind { ToolKind::Read }
    async fn run(&self, args: &Value, ctx: &ToolContext) -> ToolResult {
        let Some(pattern) = arg_str(args, "pattern") else {
            return ToolResult::err("missing required parameter 'pattern'");
        };
        let path = arg_str(args, "path").unwrap_or_else(|| ".".to_string());
        let case_insensitive = arg_bool(args, "case_insensitive", false);
        let literal = arg_bool(args, "literal", false);
        let max_matches = arg_u32(args, "max_matches", 100).max(1);

        // Prefer ripgrep if installed; it is far faster and respects gitignore.
        if which("rg").await {
            return run_rg(&pattern, &path, case_insensitive, literal, max_matches, ctx).await;
        }
        ToolResult::err(
            "ripgrep (rg) not found on PATH. Install it or use the Phoenix Agent installer.",
        )
    }
}

/// Return true if a command exists on PATH.
async fn which(cmd: &str) -> bool {
    tokio::process::Command::new(cmd)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .await
        .is_ok()
}

async fn run_rg(
    pattern: &str,
    path: &str,
    case_insensitive: bool,
    literal: bool,
    max_matches: u32,
    ctx: &ToolContext,
) -> ToolResult {
    let resolved = ctx.workdir.join(path);
    let mut cmd = tokio::process::Command::new("rg");
    cmd.arg("--line-number")
        .arg("--no-heading")
        .arg("--color=never")
        .arg(format!("--max-count={max_matches}"))
        .current_dir(&ctx.workdir);
    if case_insensitive {
        cmd.arg("-i");
    }
    if literal {
        cmd.arg("-F");
    }
    cmd.arg("--").arg(pattern).arg(&resolved);
    match cmd.output().await {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            if stdout.is_empty() && !out.status.success() && !stderr.is_empty() {
                // rg returns exit 1 when no matches; that's not an error for us.
                if !stderr.contains("No files were searched") {
                    return ToolResult::err(stderr.to_string());
                }
            }
            if stdout.is_empty() {
                ToolResult::ok(format!("No matches for pattern '{pattern}'."))
            } else {
                ToolResult::ok(format!("Matches:\n{}", stdout.trim_end()))
            }
        }
        Err(e) => ToolResult::err(format!("run rg: {e}")),
    }
}

