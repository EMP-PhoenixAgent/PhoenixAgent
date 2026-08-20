//! System prompt construction.

use crate::agent::tools::ToolContext;
use crate::config::Mode;

/// Build the system prompt that frames the agent's identity and behavior.
///
/// This is prepended to every conversation as a `system` message. It tells the
/// model who it is, what tools it has, and how to behave.
/// - `skills` — enabled skills (name, description, body); rendered into a
///   `## Skills` section framed as methodology to apply when relevant.
/// - `context` — enabled context files (name, description, body); rendered into
///   a `## Project Context` section framed as authoritative ground truth the
///   model must respect and never contradict.
pub fn build_system_prompt(
    ctx: &ToolContext,
    tool_summaries: &[(&str, &str)],
    skills: &[(&str, &str, &str)],
    context: &[(&str, &str, &str)],
    mode: Mode,
) -> String {
    let mut s = String::new();

    s.push_str("You are Phoenix Agent, a fully-local, autonomous coding assistant.\n\n");

    s.push_str("## Identity\n");
    s.push_str("You help the user with software engineering tasks: reading, writing, and \
                editing code; searching codebases; running builds and tests; and explaining \
                your work. You are running entirely on the user's local machine. You operate \
                honestly: you state what you did, quote real output, and never fabricate \
                results or file contents.\n\n");

    s.push_str("## Environment\n");
    s.push_str(&format!("- Operating system: {}\n", ctx.os));
    s.push_str(&format!("- Working directory: {}\n", ctx.workdir.display()));
    s.push_str("- Prefer absolute paths or paths relative to the working directory.\n\n");

    s.push_str("## How to work\n");
    s.push_str("- Investigate before acting. Use `list_dir`, `read_file`, and `grep` to \
                understand the codebase before making changes.\n");
    s.push_str("- Make the smallest correct change. Prefer `edit_file` over rewriting whole \
                files with `write_file`.\n");
    s.push_str("- After making changes, verify them (e.g. run the build/tests with \
                `run_command`) and report the real outcome.\n");
    s.push_str("- If a command fails, read the error, fix the cause, and retry — don't claim \
                success without confirmation.\n");
    s.push_str("- When you need information only the user can provide, ask a concise question.\n");
    s.push_str("- When the task is complete, give a brief summary of what changed.\n\n");

    s.push_str("## Output style\n");
    s.push_str("- Be concise and direct. Use code blocks for code, commands, and file paths.\n");
    s.push_str("- Explain your reasoning briefly before acting, then act. Don't narrate every \
                trivial step.\n\n");

    s.push_str("## Tools\n");
    s.push_str("Call a tool by emitting a tool call with the function name and JSON arguments.\n");
    for (name, desc) in tool_summaries {
        s.push_str(&format!("- `{name}`: {desc}\n"));
    }

    // Skills section — only when at least one skill is enabled.
    if !skills.is_empty() {
        s.push_str("\n## Skills\n");
        s.push_str("The following skills are enabled. Apply their guidance when relevant to \
                    the task.\n\n");
        for (i, (name, description, body)) in skills.iter().enumerate() {
            if i > 0 {
                s.push_str("\n---\n\n");
            }
            s.push_str(&format!("### {name}\n"));
            if !description.is_empty() {
                s.push_str(&format!("{description}\n\n"));
            }
            s.push_str(body);
            if !body.ends_with('\n') {
                s.push('\n');
            }
        }
    }

    // Project context section — only when at least one context file is enabled.
    // Framed as authoritative ground truth, distinct from the optional-methodology
    // framing of skills.
    if !context.is_empty() {
        s.push_str("\n## Project Context\n");
        s.push_str("These are facts about the current project. Treat them as ground truth \
                    and never contradict them.\n\n");
        for (i, (name, description, body)) in context.iter().enumerate() {
            if i > 0 {
                s.push_str("\n---\n\n");
            }
            s.push_str(&format!("### {name}\n"));
            if !description.is_empty() {
                s.push_str(&format!("{description}\n\n"));
            }
            s.push_str(body);
            if !body.ends_with('\n') {
                s.push('\n');
            }
        }
    }

    // Operating mode — tells the model how to behave (on top of the approval
    // gate), so it self-limits instead of attempting disallowed actions.
    s.push_str(&format!(
        "\n## Operating Mode: {}\n{}\n",
        mode.label(),
        mode.directive()
    ));

    s.push_str("\nBegin by acknowledging the task briefly, then act.\n");
    s
}
