//! Skills: markdown knowledge files injected into the agent's system prompt.
//!
//! This module holds (a) the bundled starter-skill content seeded on first run,
//! and (b) the GitHub search/install helpers used by the Skills panel. Skills
//! themselves are pure data (rows in the `skills` table) — there is no skill
//! registry to instantiate, unlike tools.

use serde::Deserialize;
use serde::Serialize;

use crate::error::{PhoenixError, Result};

/// Bundled starter skills created on first run (or when the `skills` table is
/// empty). Each tuple is `(name, description, markdown body)`. Kept short and
/// genuinely useful so the panel isn't empty and the agent immediately benefits.
pub const STARTER_SKILLS: &[(&str, &str, &str)] = &[
    (
        "git-workflow",
        "Standard git hygiene for staged, atomic commits.",
        "When making git changes:\n\
         - Show status with `git status` before staging.\n\
         - Stage logically related changes together; avoid catch-all commits.\n\
         - Write commit messages in the imperative mood (e.g. \"Add login rate limiting\").\n\
         - Run the build/tests before committing.\n\
         - Prefer several small commits over one large one.\n",
    ),
    (
        "code-review",
        "How to review and explain existing code.",
        "When asked to review or explain code:\n\
         - Read the relevant files and surrounding context before commenting.\n\
         - Point out concrete issues (bugs, security, edge cases) with file:line references.\n\
         - Distinguish must-fix problems from style nits.\n\
         - Suggest the smallest change that addresses each issue.\n\
         - Don't speculate about code you haven't read.\n",
    ),
    (
        "rust-debugging",
        "Diagnosing Rust build/test failures.",
        "When a Rust build or test fails:\n\
         - Read the full compiler/test output; address the first error first (later ones often cascade).\n\
         - For borrow-checker errors, identify which borrow outlives the data.\n\
         - Use `cargo check` for fast feedback, `cargo build`/`cargo test` to confirm.\n\
         - Quote the real error, then the fix — never claim success without re-running.\n",
    ),
    (
        "investigate-first",
        "Always understand the codebase before editing.",
        "Before writing or editing code:\n\
         - Use `list_dir`, `grep`, and `read_file` to find the relevant code and conventions.\n\
         - Match the surrounding code's style, naming, and patterns.\n\
         - Make the smallest correct change; prefer `edit_file` over full rewrites.\n\
         - Verify the change (build/test) and report the real outcome.\n",
    ),
];

/// A single GitHub code-search hit representing a candidate skill file.
#[derive(Debug, Clone, Serialize)]
pub struct GithubSkillHit {
    /// Human-readable label, e.g. `"repo/path/to/skill.md"`.
    pub name: String,
    /// Full repo name, e.g. `"owner/repo"`.
    pub repo: String,
    /// Path within the repo, e.g. `"skills/skill.md"`.
    pub path: String,
    /// Raw file URL for fetching the content.
    pub raw_url: String,
    /// HTML URL for viewing in a browser.
    pub html_url: String,
}

// ---- GitHub API wire types ------------------------------------------------

#[derive(Debug, Deserialize)]
struct SearchResponse {
    items: Vec<SearchItem>,
}

#[derive(Debug, Deserialize)]
struct SearchItem {
    path: String,
    html_url: String,
    repository: SearchRepo,
}

#[derive(Debug, Deserialize)]
struct SearchRepo {
    full_name: String,
}

/// Search GitHub for markdown files matching the query, returning candidate
/// skill hits. Uses the unauthenticated Search API (rate-limited; ~10/min). A
/// `User-Agent` header is required by the API.
pub async fn search_github(query: &str) -> Result<Vec<GithubSkillHit>> {
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| PhoenixError::Other(format!("build http client: {e}")))?;
    // Restrict to markdown files. The Search API treats spaces as AND.
    let q = format!("{query} extension:md");
    let resp = client
        .get("https://api.github.com/search/code")
        .header("User-Agent", "phoenix-agent")
        .header("Accept", "application/vnd.github+json")
        .query(&[("q", q.as_str()), ("per_page", "20")])
        .send()
        .await
        .map_err(|e| PhoenixError::Other(format!("github search request: {e}")))?;

    if resp.status() == reqwest::StatusCode::FORBIDDEN {
        // Rate limit or abuse detection.
        return Err(PhoenixError::Other(
            "GitHub rate limit reached (unauthenticated search is ~10/min). \
             Wait a moment and try again."
                .into(),
        ));
    }
    if !resp.status().is_success() {
        return Err(PhoenixError::Other(format!(
            "github search failed: HTTP {}",
            resp.status()
        )));
    }

    let parsed: SearchResponse = resp
        .json()
        .await
        .map_err(|e| PhoenixError::Other(format!("github search parse: {e}")))?;

    Ok(parsed
        .items
        .into_iter()
        .map(|item| {
            let repo = item.repository.full_name;
            GithubSkillHit {
                name: format!("{}/{}", repo, item.path),
                repo: repo.clone(),
                // The code-search html_url points at the file in the web UI; derive
                // the raw URL by swapping the host. We can't read the sha from the
                // search payload, so use the default branch raw endpoint.
                raw_url: html_to_raw(&repo, &item.path),
                path: item.path,
                html_url: item.html_url,
            }
        })
        .collect())
}

/// Convert a GitHub web file URL to a `raw.githubusercontent.com` URL on the
/// default branch. The code-search API doesn't expose the sha, so this targets
/// `HEAD`/default-branch raw content (good enough for installing a skill).
fn html_to_raw(repo: &str, path: &str) -> String {
    // Use the default-branch raw endpoint via the `/HEAD/` shorthand supported
    // by raw.githubusercontent.com.
    format!("https://raw.githubusercontent.com/{repo}/HEAD/{path}")
}

/// Fetch the raw text content of a file at a `raw.githubusercontent.com` URL.
pub async fn fetch_raw(raw_url: &str) -> Result<String> {
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| PhoenixError::Other(format!("build http client: {e}")))?;
    let resp = client
        .get(raw_url)
        .header("User-Agent", "phoenix-agent")
        .send()
        .await
        .map_err(|e| PhoenixError::Other(format!("fetch raw file: {e}")))?;
    if !resp.status().is_success() {
        return Err(PhoenixError::Other(format!(
            "fetch raw file failed: HTTP {}",
            resp.status()
        )));
    }
    resp.text()
        .await
        .map_err(|e| PhoenixError::Other(format!("read raw file: {e}")))
}

/// Derive a short skill name from a file path (last path segment, minus the
/// `.md` extension).
pub fn name_from_path(path: &str) -> String {
    path.rsplit('/')
        .next()
        .map(|f| f.trim_end_matches(".md").to_string())
        .unwrap_or_else(|| "skill".to_string())
}
