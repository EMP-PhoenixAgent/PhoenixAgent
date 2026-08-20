//! Agent orchestration: the ReAct reasoning loop and its runtime state.
//!
//! The agent loop is a generator-style design: it runs on a background task
//! and streams [`AgentEvent`]s (assistant text deltas, tool requests needing
//! approval, tool results, and completion) back to the TUI through a channel.
//! The TUI renders each event as it arrives.

pub mod tools;
pub mod tool_seeds;
pub mod prompt;
pub mod runtime;
pub mod skills;
pub mod mcp;

pub use runtime::{AgentEvent, AgentHandle, AgentRuntime, Command};
pub use skills::{GithubSkillHit, STARTER_SKILLS};
pub use tool_seeds::{starter_tools, SeedTool};
