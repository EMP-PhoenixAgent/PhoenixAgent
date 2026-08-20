//! Filesystem tools: read_file, write_file, edit_file, list_dir.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::Path;

use super::{arg_bool, arg_str, Tool, ToolContext, ToolResult};
use crate::config::ToolKind;

/// Read a file's contents (UTF-8). Read-only.
pub struct ReadFile;
#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str { "read_file" }
    fn description(&self) -> &str {
        "Read the complete contents of a UTF-8 text file. Use an absolute path or a path relative to the project working directory."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to the file to read." }
            },
            "required": ["path"]
        })
    }
    fn kind(&self) -> ToolKind { ToolKind::Read }
    async fn run(&self, args: &Value, ctx: &ToolContext) -> ToolResult {
        let Some(path) = arg_str(args, "path") else {
            return ToolResult::err("missing required parameter 'path'");
        };
        let resolved = resolve(&path, &ctx.workdir);
        match tokio::fs::read_to_string(&resolved).await {
            Ok(contents) => {
                let lines = contents.matches('\n').count() + 1;
                ToolResult::ok(format!(
                    "File: {} ({} lines, {} bytes)\n```\n{contents}\n```",
                    resolved.display(),
                    lines,
                    contents.len()
                ))
            }
            Err(e) => ToolResult::err(format!("read {}: {e}", resolved.display())),
        }
    }
}

/// Write (create or overwrite) a file. Mutating — approval-gated.
pub struct WriteFile;
#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &str { "write_file" }
    fn description(&self) -> &str {
        "Create or overwrite a file with the given content. Creates parent directories if needed."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path":    { "type": "string", "description": "Path to the file to write." },
                "content": { "type": "string", "description": "Full new contents of the file." }
            },
            "required": ["path", "content"]
        })
    }
    fn kind(&self) -> ToolKind { ToolKind::Write }
    async fn run(&self, args: &Value, ctx: &ToolContext) -> ToolResult {
        let (Some(path), Some(content)) = (arg_str(args, "path"), arg_str(args, "content")) else {
            return ToolResult::err("missing required parameters 'path' and 'content'");
        };
        let resolved = resolve(&path, &ctx.workdir);
        if let Some(parent) = resolved.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return ToolResult::err(format!("create dirs: {e}"));
            }
        }
        match tokio::fs::write(&resolved, &content).await {
            Ok(_) => ToolResult::ok(format!(
                "Wrote {} bytes to {}",
                content.len(),
                resolved.display()
            )),
            Err(e) => ToolResult::err(format!("write {}: {e}", resolved.display())),
        }
    }
}

/// Replace a unique string within a file. Mutating — approval-gated.
pub struct EditFile;
#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str { "edit_file" }
    fn description(&self) -> &str {
        "Replace one occurrence of `find` with `replace` in a file. Fails if `find` is absent or appears more than once unless `replace_all` is true."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path":        { "type": "string" },
                "find":        { "type": "string", "description": "Exact text to locate." },
                "replace":     { "type": "string", "description": "Replacement text." },
                "replace_all": { "type": "boolean", "default": false }
            },
            "required": ["path", "find", "replace"]
        })
    }
    fn kind(&self) -> ToolKind { ToolKind::Write }
    async fn run(&self, args: &Value, ctx: &ToolContext) -> ToolResult {
        let (Some(path), Some(find), Some(replace)) =
            (arg_str(args, "path"), arg_str(args, "find"), arg_str(args, "replace"))
        else {
            return ToolResult::err("missing required parameters 'path', 'find', 'replace'");
        };
        let all = arg_bool(args, "replace_all", false);
        let resolved = resolve(&path, &ctx.workdir);
        let content = match tokio::fs::read_to_string(&resolved).await {
            Ok(c) => c,
            Err(e) => return ToolResult::err(format!("read {}: {e}", resolved.display())),
        };
        let count = content.matches(&find).count();
        if count == 0 {
            return ToolResult::err(format!(
                "`find` text not present in {}",
                resolved.display()
            ));
        }
        if count > 1 && !all {
            return ToolResult::err(format!(
                "`find` text appears {count} times in {}; set replace_all=true or use a more specific find string",
                resolved.display()
            ));
        }
        let new = if all {
            content.replace(&find, &replace)
        } else {
            content.replacen(&find, &replace, 1)
        };
        if let Err(e) = tokio::fs::write(&resolved, &new).await {
            return ToolResult::err(format!("write {}: {e}", resolved.display()));
        }
        ToolResult::ok(format!(
            "Edited {} ({} replacement{})",
            resolved.display(),
            if all { count } else { 1 },
            if (all && count > 1) || (!all) { "s" } else { "" }
        ))
    }
}

/// List a directory's entries. Read-only.
pub struct ListDir;
#[async_trait]
impl Tool for ListDir {
    fn name(&self) -> &str { "list_dir" }
    fn description(&self) -> &str {
        "List the entries of a directory, marking directories with a trailing slash."
    }
    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path; defaults to the working directory." }
            }
        })
    }
    fn kind(&self) -> ToolKind { ToolKind::Read }
    async fn run(&self, args: &Value, ctx: &ToolContext) -> ToolResult {
        let path = arg_str(args, "path").unwrap_or_else(|| ".".into());
        let resolved = resolve(&path, &ctx.workdir);
        let mut entries = match tokio::fs::read_dir(&resolved).await {
            Ok(rd) => rd,
            Err(e) => return ToolResult::err(format!("read_dir {}: {e}", resolved.display())),
        };
        let mut names: Vec<String> = Vec::new();
        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();
            let suffix = if entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false) {
                "/"
            } else {
                ""
            };
            names.push(format!("{name}{suffix}"));
        }
        names.sort();
        ToolResult::ok(format!(
            "{} ({} entries):\n{}",
            resolved.display(),
            names.len(),
            names.join("\n")
        ))
    }
}

/// Resolve a user-supplied path against the working directory.
fn resolve(path: &str, workdir: &Path) -> std::path::PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        workdir.join(p)
    }
}
