//! The memory store — queries over sessions, messages, projects, and notes.
//!
//! This is the "memorize everything" layer: every turn of every session is
//! persisted here, queryable later. The store is a thin wrapper over a
//! [`rusqlite::Connection`]; callers own the connection via [`MemoryStore`].

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};

use crate::error::Result;
use crate::model::{ChatMessage, ChatRole};

/// Owns the encrypted connection and exposes memory operations.
pub struct MemoryStore {
    conn: Connection,
}

/// Lightweight session info for the resume / session-list UI.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSummary {
    pub id: i64,
    pub title: String,
    pub model: String,
    pub created_at: String,
    pub updated_at: String,
    pub message_count: i64,
}

/// A named bundle of agent behavior settings (Science Workbench profile).
///
/// The active model and working directory are intentionally NOT part of a
/// profile — they are independent sidebar selectors persisted in `settings`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Profile {
    pub id: i64,
    pub name: String,
    /// `"all" | "writes_only" | "never"` — stored as TEXT, mirrored from
    /// [`crate::config::ApprovalPolicy`].
    pub approval_policy: String,
    pub max_iterations: i64,
    pub context_window: i64,
    pub is_default: bool,
    pub created_at: String,
    pub updated_at: String,
}

/// A markdown knowledge file injected into the agent's system prompt.
///
/// Skills are either authored locally (`source = "local"`) or installed from a
/// GitHub raw file (`source = "github"`). Per-profile enable/disable lives in
/// the `profile_skills` join table; a profile with no rows there inherits the
/// `enabled_global` default.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Skill {
    pub id: i64,
    pub name: String,
    pub description: String,
    /// Markdown body injected into the prompt's `## Skills` section.
    pub body: String,
    /// `"local" | "github"`.
    pub source: String,
    /// Raw file URL when `source = "github"`, else `None`.
    pub source_url: Option<String>,
    pub enabled_global: bool,
    pub created_at: String,
    pub updated_at: String,
}

/// A predefined specialist sub-agent the main agent can delegate sub-tasks to.
/// Each runs on its configured model (served by AmberCore) with its own persona.
/// (Science Workbench Panel 6.)
#[derive(Debug, Clone, serde::Serialize)]
pub struct SubAgent {
    pub id: i64,
    pub name: String,
    pub description: String,
    /// The sub-agent's system prompt — its persona/specialty instructions.
    pub persona: String,
    /// Model tag to run on (empty string = use the currently-active model).
    pub model: String,
    pub created_at: String,
    pub updated_at: String,
}

/// A skill plus its enabled state for a specific profile (for the panel UI).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileSkill {
    #[serde(flatten)]
    pub skill: Skill,
    pub enabled: bool,
}

/// A user-installable executable tool (Panel 3). The `script_body` is run with
/// `interpreter`; the model's arguments are fed as JSON on stdin.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ToolRow {
    pub id: i64,
    pub name: String,
    pub description: String,
    /// `"python" | "node" | "sh" | "powershell" | <program>`.
    pub interpreter: String,
    /// Script source (written to a temp file at run time).
    pub script_body: String,
    /// JSON Schema (object) describing the tool's parameters, as TEXT.
    pub params_schema: String,
    /// `"read" | "write"` — drives the approval gate.
    pub tool_kind: String,
    /// `"local" | "github"`.
    pub source: String,
    pub source_url: Option<String>,
    pub enabled_global: bool,
    pub created_at: String,
    pub updated_at: String,
}

/// A tool plus its enabled state for a specific profile (for the panel UI).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileTool {
    #[serde(flatten)]
    pub tool: ToolRow,
    pub enabled: bool,
}

/// A declarative project-knowledge file (Panel 4). Injected into the prompt as
/// authoritative ground truth the model must respect.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextFile {
    pub id: i64,
    pub name: String,
    pub description: String,
    /// Markdown body injected under the `## Project Context` section.
    pub body: String,
    pub enabled_global: bool,
    pub created_at: String,
    pub updated_at: String,
}

/// A context file plus its enabled state for a specific profile (panel UI).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileContext {
    #[serde(flatten)]
    pub context: ContextFile,
    pub enabled: bool,
}

/// An external MCP server connection (Panel 5 / "memory"). When enabled for a
/// profile, the agent spawns/contacts the server and registers its tools.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemorySource {
    pub id: i64,
    pub name: String,
    pub description: String,
    /// Transport: `"stdio"` (spawn + JSON-RPC over stdin/stdout) or `"http"`.
    pub transport: String,
    /// Executable path (stdio) or base URL (http).
    pub command: String,
    /// JSON array of string args (stdio) / extra config. Stored verbatim.
    pub args_json: String,
    pub enabled_global: bool,
    pub created_at: String,
    pub updated_at: String,
}

/// A memory source plus its enabled state for a specific profile (panel UI).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ProfileMemory {
    #[serde(flatten)]
    pub source: MemorySource,
    pub enabled: bool,
}

/// A registered cloud provider (Models panel red box). The API key lives in the
/// encrypted DB; `list_providers` returns it **masked** (never the cleartext)
/// for the panel list. Use [`MemoryStore::get_provider_key`] for the cleartext.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Provider {
    pub id: i64,
    pub name: String,
    pub base_url: String,
    /// Masked key, e.g. `••••1234`. Safe to send to the webview.
    pub api_key_masked: String,
    /// Protocol family; `"openai"` for v1 (the only family supported so far).
    pub kind: String,
    pub created_at: String,
    pub updated_at: String,
}

impl MemoryStore {
    /// Wrap an existing (already-unlocked) connection.
    pub fn new(conn: Connection) -> Self {
        Self { conn }
    }

    /// Borrow the underlying connection (for ad-hoc queries / doctor).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    // ---- projects -------------------------------------------------------

    /// Get-or-create a project row for the given path, returning its id.
    pub fn ensure_project(&self, path: &str, name: Option<&str>) -> Result<i64> {
        let existing: Option<i64> = self
            .conn
            .query_row("SELECT id FROM projects WHERE path = ?1", params![path], |row| {
                row.get(0)
            })
            .optional()?;
        if let Some(id) = existing {
            return Ok(id);
        }
        let name = name.unwrap_or_else(|| {
            std::path::Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("project")
        });
        self.conn.execute(
            "INSERT INTO projects (name, path) VALUES (?1, ?2)",
            params![name, path],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ---- sessions -------------------------------------------------------

    /// Start a new session. Returns the session id.
    pub fn start_session(&self, project_id: Option<i64>, model: &str, title: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO sessions (project_id, model, title) VALUES (?1, ?2, ?3)",
            params![project_id, model, title],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// List sessions, newest first.
    pub fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.id, s.title, s.model, s.created_at, s.updated_at,
                    (SELECT count(*) FROM messages m WHERE m.session_id = s.id) AS n
             FROM sessions s
             ORDER BY s.updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(SessionSummary {
                id: row.get(0)?,
                title: row.get(1)?,
                model: row.get(2)?,
                created_at: row.get(3)?,
                updated_at: row.get(4)?,
                message_count: row.get(5)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // ---- messages -------------------------------------------------------

    /// Append a raw chat message row. Returns the new message id.
    pub fn append_message(
        &self,
        session_id: i64,
        message: &ChatMessage,
        model: Option<&str>,
        tokens_in: Option<i64>,
        tokens_out: Option<i64>,
    ) -> Result<i64> {
        let role = message.role.as_str();
        let tool_calls = match &message.tool_calls {
            Some(tc) => Some(serde_json::to_string(tc)?),
            None => None,
        };
        let tool_name = match message.role {
            ChatRole::Tool => message.tool_name.as_deref(),
            _ => None,
        };
        self.conn.execute(
            "INSERT INTO messages
                (session_id, role, content, tool_calls, tool_name, model, tokens_in, tokens_out)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                session_id,
                role,
                message.content,
                tool_calls,
                tool_name,
                model,
                tokens_in,
                tokens_out
            ],
        )?;
        let id = self.conn.last_insert_rowid();
        // Bump session updated_at.
        self.conn.execute(
            "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
            params![now_iso(), session_id],
        )?;
        Ok(id)
    }

    /// Load the last `limit` messages of a session (chronological order).
    pub fn load_messages(&self, session_id: i64, limit: u32) -> Result<Vec<ChatMessage>> {
        let mut stmt = self.conn.prepare(
            "SELECT role, content, tool_calls, tool_name
             FROM messages
             WHERE session_id = ?1
             ORDER BY id DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![session_id, limit], |row| {
            let role: String = row.get(0)?;
            let content: String = row.get(1).unwrap_or_default();
            let tool_calls: Option<String> = row.get(2)?;
            let tool_name: Option<String> = row.get(3)?;
            let mut msg = ChatMessage {
                role: ChatRole::parse(&role),
                content,
                tool_calls: None,
                tool_name,
            };
            if let Some(tc) = tool_calls {
                if !tc.is_empty() {
                    msg.tool_calls = Some(serde_json::from_str(&tc).unwrap_or_default());
                }
            }
            Ok(msg)
        })?;
        let mut out: Vec<ChatMessage> = Vec::new();
        for r in rows {
            out.push(r?);
        }
        out.reverse(); // back to chronological
        Ok(out)
    }

    // ---- notes ----------------------------------------------------------

    // ---- profiles (Science Workbench) ----------------------------------

    /// Ensure a "Default" profile exists. On a fresh or just-upgraded DB the
    /// `profiles` table is empty, so we insert one row with the built-in
    /// defaults and `is_default = 1`. Returns the default profile id.
    pub fn ensure_default_profile(&self) -> Result<i64> {
        let existing: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM profiles WHERE is_default = 1 LIMIT 1",
                [],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(id) = existing {
            return Ok(id);
        }
        // Fall back to any profile if none is flagged default yet.
        let any: Option<i64> = self
            .conn
            .query_row("SELECT id FROM profiles LIMIT 1", [], |row| row.get(0))
            .optional()?;
        if let Some(id) = any {
            return Ok(id);
        }
        self.conn.execute(
            "INSERT INTO profiles (name, approval_policy, max_iterations, context_window, is_default)
             VALUES ('Default', 'writes_only', 25, 50, 1)",
            [],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// List all profiles, default first then alphabetical.
    pub fn list_profiles(&self) -> Result<Vec<Profile>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, name, approval_policy, max_iterations, context_window,
                    is_default, created_at, updated_at
             FROM profiles
             ORDER BY is_default DESC, name ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok(Profile {
                id: row.get(0)?,
                name: row.get(1)?,
                approval_policy: row.get(2)?,
                max_iterations: row.get(3)?,
                context_window: row.get(4)?,
                is_default: row.get::<_, i64>(5)? != 0,
                created_at: row.get(6)?,
                updated_at: row.get(7)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Fetch a single profile by id.
    pub fn get_profile(&self, id: i64) -> Result<Option<Profile>> {
        let row = self
            .conn
            .query_row(
                "SELECT id, name, approval_policy, max_iterations, context_window,
                        is_default, created_at, updated_at
                 FROM profiles WHERE id = ?1",
                params![id],
                |row| {
                    Ok(Profile {
                        id: row.get(0)?,
                        name: row.get(1)?,
                        approval_policy: row.get(2)?,
                        max_iterations: row.get(3)?,
                        context_window: row.get(4)?,
                        is_default: row.get::<_, i64>(5)? != 0,
                        created_at: row.get(6)?,
                        updated_at: row.get(7)?,
                    })
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Create a new profile with the given behavior settings.
    pub fn create_profile(
        &self,
        name: &str,
        approval_policy: &str,
        max_iterations: u32,
        context_window: u32,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO profiles (name, approval_policy, max_iterations, context_window)
             VALUES (?1, ?2, ?3, ?4)",
            params![name, approval_policy, max_iterations, context_window],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    // ---- settings KV (active profile, workdir, ...) --------------------

    /// Read a value from the `settings` table, if present.
    pub fn get_setting(&self, key: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT value FROM settings WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    /// Upsert a value into the `settings` table.
    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Get the active profile id, falling back to the default if unset.
    pub fn get_active_profile_id(&self) -> Result<Option<i64>> {
        if let Some(v) = self.get_setting("active_profile_id")? {
            if let Ok(id) = v.parse::<i64>() {
                // Confirm it still exists; if deleted, fall through to default.
                if self.get_profile(id)?.is_some() {
                    return Ok(Some(id));
                }
            }
        }
        // Unset or stale — point at the default and return it.
        let default_id = self.ensure_default_profile()?;
        let _ = self.set_active_profile_id(default_id);
        Ok(Some(default_id))
    }

    /// Set the active profile id (persisted across restarts).
    pub fn set_active_profile_id(&self, id: i64) -> Result<()> {
        self.set_setting("active_profile_id", &id.to_string())
    }

    /// Get the persisted working directory, if any.
    pub fn get_workdir(&self) -> Result<Option<String>> {
        self.get_setting("workdir")
    }

    /// Persist the working directory.
    pub fn set_workdir(&self, path: &str) -> Result<()> {
        self.set_setting("workdir", path)
    }

    // ---- TOTP 2FA (settings KV rows) -----------------------------------

    /// Get the configured TOTP secret (base32) and account label, if 2FA is on.
    pub fn get_totp_config(&self) -> Result<Option<(String, String)>> {
        match self.get_setting("totp_secret_b32")? {
            Some(secret) => {
                let account = self.get_setting("totp_account")?.unwrap_or_default();
                Ok(Some((secret, account)))
            }
            None => Ok(None),
        }
    }

    /// Persist a TOTP config (enables 2FA).
    pub fn set_totp_config(&self, secret_b32: &str, account: &str) -> Result<()> {
        self.set_setting("totp_secret_b32", secret_b32)?;
        self.set_setting("totp_account", account)?;
        Ok(())
    }

    /// Remove the TOTP config entirely (disables 2FA).
    pub fn clear_totp_config(&self) -> Result<()> {
        // settings table has no row-delete helper; overwrite to empty + rely on
        // get_totp_config treating empty as "unset". A real DELETE is cleaner:
        self.conn.execute("DELETE FROM settings WHERE key IN (?1, ?2, ?3)", params![
            "totp_secret_b32",
            "totp_account",
            "totp_pending_secret_b32",
        ])?;
        Ok(())
    }

    // ---- skills (Science Workbench Panel 2) ----------------------------

    /// Map a row to a [`Skill`]. Shared by all skill readers.
    fn read_skill_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Skill> {
        Ok(Skill {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            body: row.get(3)?,
            source: row.get(4)?,
            source_url: row.get(5)?,
            enabled_global: row.get::<_, i64>(6)? != 0,
            created_at: row.get(7)?,
            updated_at: row.get(8)?,
        })
    }

    const SKILL_COLUMNS: &'static str =
        "id, name, description, body, source, source_url, enabled_global, created_at, updated_at";

    /// Seed a handful of starter skills if the table is empty (first run or a
    /// freshly upgraded DB). Idempotent.
    pub fn ensure_seed_skills(&self) -> Result<()> {
        let count: i64 = self
            .conn
            .query_row("SELECT count(*) FROM skills", [], |row| row.get(0))?;
        if count > 0 {
            return Ok(());
        }
        for (name, desc, body) in crate::agent::skills::STARTER_SKILLS {
            self.conn.execute(
                "INSERT INTO skills (name, description, body, source, enabled_global)
                 VALUES (?1, ?2, ?3, 'local', 1)",
                params![name, desc, body],
            )?;
        }
        Ok(())
    }

    /// List all skills, alphabetical by name.
    pub fn list_skills(&self) -> Result<Vec<Skill>> {
        let sql = format!("SELECT {} FROM skills ORDER BY name ASC", Self::SKILL_COLUMNS);
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], Self::read_skill_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Fetch a single skill by id.
    pub fn get_skill(&self, id: i64) -> Result<Option<Skill>> {
        let sql = format!("SELECT {} FROM skills WHERE id = ?1", Self::SKILL_COLUMNS);
        let row = self
            .conn
            .query_row(&sql, params![id], Self::read_skill_row)
            .optional()?;
        Ok(row)
    }

    /// Create a new skill. Returns its id.
    pub fn create_skill(
        &self,
        name: &str,
        description: &str,
        body: &str,
        source: &str,
        source_url: Option<&str>,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO skills (name, description, body, source, source_url)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![name, description, body, source, source_url],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update a skill's editable fields (name/description/body).
    pub fn update_skill(
        &self,
        id: i64,
        name: &str,
        description: &str,
        body: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE skills SET name = ?1, description = ?2, body = ?3, updated_at = ?4
             WHERE id = ?5",
            params![name, description, body, now_iso(), id],
        )?;
        Ok(())
    }

    /// Delete a skill. `profile_skills` rows cascade automatically.
    pub fn delete_skill(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM skills WHERE id = ?1", params![id])?;
        Ok(())
    }

    // ---- sub_agents (Science Workbench Panel 6) ------------------------

    /// Map a row to a [`SubAgent`]. Shared by all sub-agent readers.
    fn read_sub_agent_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SubAgent> {
        Ok(SubAgent {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            persona: row.get(3)?,
            model: row.get(4)?,
            created_at: row.get(5)?,
            updated_at: row.get(6)?,
        })
    }

    const SUB_AGENT_COLUMNS: &'static str =
        "id, name, description, persona, model, created_at, updated_at";

    /// List all sub-agents, ordered by name.
    pub fn list_sub_agents(&self) -> Result<Vec<SubAgent>> {
        let sql = format!(
            "SELECT {} FROM sub_agents ORDER BY name ASC",
            Self::SUB_AGENT_COLUMNS
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], Self::read_sub_agent_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Create a new sub-agent. Returns its id.
    pub fn create_sub_agent(
        &self,
        name: &str,
        description: &str,
        persona: &str,
        model: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO sub_agents (name, description, persona, model)
             VALUES (?1, ?2, ?3, ?4)",
            params![name, description, persona, model],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update a sub-agent's editable fields.
    pub fn update_sub_agent(
        &self,
        id: i64,
        name: &str,
        description: &str,
        persona: &str,
        model: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE sub_agents SET name = ?1, description = ?2, persona = ?3, model = ?4,
             updated_at = ?5 WHERE id = ?6",
            params![name, description, persona, model, now_iso(), id],
        )?;
        Ok(())
    }

    /// Delete a sub-agent.
    pub fn delete_sub_agent(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM sub_agents WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Does a profile have ANY explicit `profile_skills` rows (i.e. has the
    /// user customized it)? If not, the profile inherits `enabled_global`.
    fn profile_has_custom_skills(&self, profile_id: i64) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM profile_skills WHERE profile_id = ?1",
            params![profile_id],
            |row| row.get(0),
        )?;
        Ok(n > 0)
    }

    /// List the skills enabled for a profile. If the profile has no explicit
    /// `profile_skills` rows, returns all `enabled_global = 1` skills.
    pub fn list_enabled_skills_for_profile(&self, profile_id: i64) -> Result<Vec<Skill>> {
        if self.profile_has_custom_skills(profile_id)? {
            let sql = format!(
                "SELECT {c} FROM skills s
                 JOIN profile_skills ps ON ps.skill_id = s.id
                 WHERE ps.profile_id = ?1 AND ps.enabled = 1
                 ORDER BY s.name ASC",
                c = Self::SKILL_COLUMNS
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(params![profile_id], Self::read_skill_row)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        } else {
            let sql = format!(
                "SELECT {c} FROM skills WHERE enabled_global = 1 ORDER BY name ASC",
                c = Self::SKILL_COLUMNS
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map([], Self::read_skill_row)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        }
    }

    /// List every skill with its enabled state for a profile (for the panel UI).
    /// Skills without an explicit row show the `enabled_global` default.
    pub fn list_skills_for_profile(&self, profile_id: i64) -> Result<Vec<ProfileSkill>> {
        let customized = self.profile_has_custom_skills(profile_id)?;
        // Use a parameterized SQL that works for both cases: bind the profile id
        // when customized, or -1 (matches nothing) when not, so the LEFT JOIN
        // yields the enabled_global default for every skill.
        let sql = format!(
            "SELECT {c},
                    CASE WHEN ?1 = -1 THEN s.enabled_global
                         ELSE COALESCE(ps.enabled, s.enabled_global)
                    END AS enabled
             FROM skills s
             LEFT JOIN profile_skills ps
               ON ps.skill_id = s.id AND ps.profile_id = ?1
             ORDER BY s.name ASC",
            c = Self::SKILL_COLUMNS
        );
        let bind_id = if customized { profile_id } else { -1 };
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![bind_id], |row| {
            let skill = Self::read_skill_row(row)?;
            let enabled = row.get::<_, i64>(9)? != 0;
            Ok(ProfileSkill { skill, enabled })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Set a skill's enabled state for a profile (upsert into `profile_skills`,
    /// which marks the profile as customized).
    pub fn set_skill_enabled_for_profile(
        &self,
        profile_id: i64,
        skill_id: i64,
        enabled: bool,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO profile_skills (profile_id, skill_id, enabled)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(profile_id, skill_id) DO UPDATE SET enabled = excluded.enabled",
            params![profile_id, skill_id, enabled as i64],
        )?;
        Ok(())
    }

    // ---- tools (Science Workbench Panel 3) -----------------------------

    /// Map a row to a [`ToolRow`]. Shared by all tool readers.
    fn read_tool_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ToolRow> {
        Ok(ToolRow {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            interpreter: row.get(3)?,
            script_body: row.get(4)?,
            params_schema: row.get(5)?,
            tool_kind: row.get(6)?,
            source: row.get(7)?,
            source_url: row.get(8)?,
            enabled_global: row.get::<_, i64>(9)? != 0,
            created_at: row.get(10)?,
            updated_at: row.get(11)?,
        })
    }

    const TOOL_COLUMNS: &'static str = "id, name, description, interpreter, script_body, \
        params_schema, tool_kind, source, source_url, enabled_global, created_at, updated_at";

    /// Seed the bundled starter tools (scientific/teaching utilities) if the
    /// `tools` table is empty (first run or freshly upgraded DB). Idempotent —
    /// re-running on a populated table is a no-op, so user edits/installs are
    /// never clobbered.
    pub fn ensure_seed_tools(&self) -> Result<()> {
        let count: i64 = self
            .conn
            .query_row("SELECT count(*) FROM tools", [], |row| row.get(0))?;
        if count > 0 {
            return Ok(());
        }
        for t in crate::agent::tool_seeds::starter_tools() {
            self.conn.execute(
                "INSERT INTO tools (name, description, interpreter, script_body, params_schema,
                                    tool_kind, source, enabled_global)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'local', 1)",
                params![
                    t.name,
                    t.description,
                    t.interpreter,
                    t.script_body,
                    t.params_schema,
                    t.tool_kind,
                ],
            )?;
        }
        Ok(())
    }

    /// List all tools, alphabetical by name.
    pub fn list_tools(&self) -> Result<Vec<ToolRow>> {
        let sql = format!("SELECT {} FROM tools ORDER BY name ASC", Self::TOOL_COLUMNS);
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], Self::read_tool_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Fetch a single tool by id.
    pub fn get_tool(&self, id: i64) -> Result<Option<ToolRow>> {
        let sql = format!("SELECT {} FROM tools WHERE id = ?1", Self::TOOL_COLUMNS);
        let row = self
            .conn
            .query_row(&sql, params![id], Self::read_tool_row)
            .optional()?;
        Ok(row)
    }

    /// Create a new tool. Returns its id.
    #[allow(clippy::too_many_arguments)]
    pub fn create_tool(
        &self,
        name: &str,
        description: &str,
        interpreter: &str,
        script_body: &str,
        params_schema: &str,
        tool_kind: &str,
        source: &str,
        source_url: Option<&str>,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO tools (name, description, interpreter, script_body, params_schema,
                                tool_kind, source, source_url)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![name, description, interpreter, script_body, params_schema, tool_kind, source, source_url],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update a tool's editable fields.
    pub fn update_tool(
        &self,
        id: i64,
        name: &str,
        description: &str,
        interpreter: &str,
        script_body: &str,
        params_schema: &str,
        tool_kind: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE tools SET name = ?1, description = ?2, interpreter = ?3, script_body = ?4,
                params_schema = ?5, tool_kind = ?6, updated_at = ?7
             WHERE id = ?8",
            params![name, description, interpreter, script_body, params_schema, tool_kind, now_iso(), id],
        )?;
        Ok(())
    }

    /// Delete a tool. `profile_tools` rows cascade automatically.
    pub fn delete_tool(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM tools WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Does a profile have ANY explicit `profile_tools` rows (customized)?
    fn profile_has_custom_tools(&self, profile_id: i64) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM profile_tools WHERE profile_id = ?1",
            params![profile_id],
            |row| row.get(0),
        )?;
        Ok(n > 0)
    }

    /// List the tools enabled for a profile. If the profile has no explicit
    /// `profile_tools` rows, returns all `enabled_global = 1` tools.
    pub fn list_enabled_tools_for_profile(&self, profile_id: i64) -> Result<Vec<ToolRow>> {
        if self.profile_has_custom_tools(profile_id)? {
            let sql = format!(
                "SELECT {c} FROM tools t
                 JOIN profile_tools pt ON pt.tool_id = t.id
                 WHERE pt.profile_id = ?1 AND pt.enabled = 1
                 ORDER BY t.name ASC",
                c = Self::TOOL_COLUMNS
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(params![profile_id], Self::read_tool_row)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        } else {
            let sql = format!(
                "SELECT {c} FROM tools WHERE enabled_global = 1 ORDER BY name ASC",
                c = Self::TOOL_COLUMNS
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map([], Self::read_tool_row)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        }
    }

    /// List every tool with its enabled state for a profile (for the panel UI).
    pub fn list_tools_for_profile(&self, profile_id: i64) -> Result<Vec<ProfileTool>> {
        let customized = self.profile_has_custom_tools(profile_id)?;
        let sql = format!(
            "SELECT {c},
                    CASE WHEN ?1 = -1 THEN t.enabled_global
                         ELSE COALESCE(pt.enabled, t.enabled_global)
                    END AS enabled
             FROM tools t
             LEFT JOIN profile_tools pt
               ON pt.tool_id = t.id AND pt.profile_id = ?1
             ORDER BY t.name ASC",
            c = Self::TOOL_COLUMNS
        );
        let bind_id = if customized { profile_id } else { -1 };
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![bind_id], |row| {
            let tool = Self::read_tool_row(row)?;
            let enabled = row.get::<_, i64>(12)? != 0;
            Ok(ProfileTool { tool, enabled })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Set a tool's enabled state for a profile (upsert).
    pub fn set_tool_enabled_for_profile(
        &self,
        profile_id: i64,
        tool_id: i64,
        enabled: bool,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO profile_tools (profile_id, tool_id, enabled)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(profile_id, tool_id) DO UPDATE SET enabled = excluded.enabled",
            params![profile_id, tool_id, enabled as i64],
        )?;
        Ok(())
    }

    // ---- context (Science Workbench Panel 4) --------------------------

    /// Map a row to a [`ContextFile`]. Shared by all context readers.
    fn read_context_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ContextFile> {
        Ok(ContextFile {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            body: row.get(3)?,
            enabled_global: row.get::<_, i64>(4)? != 0,
            created_at: row.get(5)?,
            updated_at: row.get(6)?,
        })
    }

    const CONTEXT_COLUMNS: &'static str =
        "id, name, description, body, enabled_global, created_at, updated_at";

    /// List all context files, alphabetical by name.
    pub fn list_context(&self) -> Result<Vec<ContextFile>> {
        let sql = format!(
            "SELECT {} FROM context_files ORDER BY name ASC",
            Self::CONTEXT_COLUMNS
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], Self::read_context_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Fetch a single context file by id.
    pub fn get_context(&self, id: i64) -> Result<Option<ContextFile>> {
        let sql = format!(
            "SELECT {} FROM context_files WHERE id = ?1",
            Self::CONTEXT_COLUMNS
        );
        let row = self
            .conn
            .query_row(&sql, params![id], Self::read_context_row)
            .optional()?;
        Ok(row)
    }

    /// Create a new context file. Returns its id.
    pub fn create_context(&self, name: &str, description: &str, body: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO context_files (name, description, body)
             VALUES (?1, ?2, ?3)",
            params![name, description, body],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update a context file's editable fields.
    pub fn update_context(
        &self,
        id: i64,
        name: &str,
        description: &str,
        body: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE context_files SET name = ?1, description = ?2, body = ?3, updated_at = ?4
             WHERE id = ?5",
            params![name, description, body, now_iso(), id],
        )?;
        Ok(())
    }

    /// Delete a context file. `profile_context` rows cascade automatically.
    pub fn delete_context(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM context_files WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Does a profile have ANY explicit `profile_context` rows (customized)?
    fn profile_has_custom_context(&self, profile_id: i64) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM profile_context WHERE profile_id = ?1",
            params![profile_id],
            |row| row.get(0),
        )?;
        Ok(n > 0)
    }

    /// List the context files enabled for a profile. If the profile has no
    /// explicit `profile_context` rows, returns all `enabled_global = 1`.
    pub fn list_enabled_context_for_profile(&self, profile_id: i64) -> Result<Vec<ContextFile>> {
        if self.profile_has_custom_context(profile_id)? {
            let sql = format!(
                "SELECT {c} FROM context_files cf
                 JOIN profile_context pc ON pc.context_id = cf.id
                 WHERE pc.profile_id = ?1 AND pc.enabled = 1
                 ORDER BY cf.name ASC",
                c = Self::CONTEXT_COLUMNS
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(params![profile_id], Self::read_context_row)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        } else {
            let sql = format!(
                "SELECT {c} FROM context_files WHERE enabled_global = 1 ORDER BY name ASC",
                c = Self::CONTEXT_COLUMNS
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map([], Self::read_context_row)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        }
    }

    /// List every context file with its enabled state for a profile (panel UI).
    pub fn list_context_for_profile(&self, profile_id: i64) -> Result<Vec<ProfileContext>> {
        let customized = self.profile_has_custom_context(profile_id)?;
        let sql = format!(
            "SELECT {c},
                    CASE WHEN ?1 = -1 THEN cf.enabled_global
                         ELSE COALESCE(pc.enabled, cf.enabled_global)
                    END AS enabled
             FROM context_files cf
             LEFT JOIN profile_context pc
               ON pc.context_id = cf.id AND pc.profile_id = ?1
             ORDER BY cf.name ASC",
            c = Self::CONTEXT_COLUMNS
        );
        let bind_id = if customized { profile_id } else { -1 };
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![bind_id], |row| {
            let context = Self::read_context_row(row)?;
            let enabled = row.get::<_, i64>(7)? != 0;
            Ok(ProfileContext { context, enabled })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Set a context file's enabled state for a profile (upsert).
    pub fn set_context_enabled_for_profile(
        &self,
        profile_id: i64,
        context_id: i64,
        enabled: bool,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO profile_context (profile_id, context_id, enabled)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(profile_id, context_id) DO UPDATE SET enabled = excluded.enabled",
            params![profile_id, context_id, enabled as i64],
        )?;
        Ok(())
    }

    // ---- memory (Science Workbench Panel 5 — MCP connections) ----------

    /// Map a row to a [`MemorySource`]. Shared by all memory readers.
    fn read_memory_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemorySource> {
        Ok(MemorySource {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            transport: row.get(3)?,
            command: row.get(4)?,
            args_json: row.get(5)?,
            enabled_global: row.get::<_, i64>(6)? != 0,
            created_at: row.get(7)?,
            updated_at: row.get(8)?,
        })
    }

    const MEMORY_COLUMNS: &'static str = "id, name, description, transport, command, \
                                          args_json, enabled_global, created_at, updated_at";

    /// List all memory sources, alphabetical by name.
    pub fn list_memory(&self) -> Result<Vec<MemorySource>> {
        let sql = format!(
            "SELECT {} FROM memory_sources ORDER BY name ASC",
            Self::MEMORY_COLUMNS
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], Self::read_memory_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Fetch a single memory source by id.
    pub fn get_memory(&self, id: i64) -> Result<Option<MemorySource>> {
        let sql = format!(
            "SELECT {} FROM memory_sources WHERE id = ?1",
            Self::MEMORY_COLUMNS
        );
        let row = self
            .conn
            .query_row(&sql, params![id], Self::read_memory_row)
            .optional()?;
        Ok(row)
    }

    /// Create a new memory source. Returns its id.
    pub fn create_memory(
        &self,
        name: &str,
        description: &str,
        transport: &str,
        command: &str,
        args_json: &str,
    ) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO memory_sources (name, description, transport, command, args_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![name, description, transport, command, args_json],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update a memory source's editable fields.
    pub fn update_memory(
        &self,
        id: i64,
        name: &str,
        description: &str,
        transport: &str,
        command: &str,
        args_json: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE memory_sources SET name = ?1, description = ?2, transport = ?3, \
                                      command = ?4, args_json = ?5, updated_at = ?6
             WHERE id = ?7",
            params![name, description, transport, command, args_json, now_iso(), id],
        )?;
        Ok(())
    }

    /// Delete a memory source. `profile_memory` rows cascade automatically.
    pub fn delete_memory(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM memory_sources WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Does a profile have ANY explicit `profile_memory` rows (customized)?
    fn profile_has_custom_memory(&self, profile_id: i64) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM profile_memory WHERE profile_id = ?1",
            params![profile_id],
            |row| row.get(0),
        )?;
        Ok(n > 0)
    }

    /// List the memory sources enabled for a profile. If the profile has no
    /// explicit `profile_memory` rows, returns all `enabled_global = 1`.
    pub fn list_enabled_memory_for_profile(&self, profile_id: i64) -> Result<Vec<MemorySource>> {
        if self.profile_has_custom_memory(profile_id)? {
            let sql = format!(
                "SELECT {c} FROM memory_sources ms
                 JOIN profile_memory pm ON pm.memory_id = ms.id
                 WHERE pm.profile_id = ?1 AND pm.enabled = 1
                 ORDER BY ms.name ASC",
                c = Self::MEMORY_COLUMNS
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(params![profile_id], Self::read_memory_row)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        } else {
            let sql = format!(
                "SELECT {c} FROM memory_sources WHERE enabled_global = 1 ORDER BY name ASC",
                c = Self::MEMORY_COLUMNS
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map([], Self::read_memory_row)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r?);
            }
            Ok(out)
        }
    }

    /// List every memory source with its enabled state for a profile (panel UI).
    pub fn list_memory_for_profile(&self, profile_id: i64) -> Result<Vec<ProfileMemory>> {
        let customized = self.profile_has_custom_memory(profile_id)?;
        let sql = format!(
            "SELECT {c},
                    CASE WHEN ?1 = -1 THEN ms.enabled_global
                         ELSE COALESCE(pm.enabled, ms.enabled_global)
                    END AS enabled
             FROM memory_sources ms
             LEFT JOIN profile_memory pm
               ON pm.memory_id = ms.id AND pm.profile_id = ?1
             ORDER BY ms.name ASC",
            c = Self::MEMORY_COLUMNS
        );
        let bind_id = if customized { profile_id } else { -1 };
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params![bind_id], |row| {
            let source = Self::read_memory_row(row)?;
            let enabled = row.get::<_, i64>(9)? != 0;
            Ok(ProfileMemory { source, enabled })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Set a memory source's enabled state for a profile (upsert).
    pub fn set_memory_enabled_for_profile(
        &self,
        profile_id: i64,
        memory_id: i64,
        enabled: bool,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO profile_memory (profile_id, memory_id, enabled)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(profile_id, memory_id) DO UPDATE SET enabled = excluded.enabled",
            params![profile_id, memory_id, enabled as i64],
        )?;
        Ok(())
    }

    // ---- providers (Models panel red box — cloud API registry) --------

    /// Render a stored key as a masked preview (last 4 chars), for the panel list.
    fn mask_key(key: &str) -> String {
        let trimmed = key.trim();
        let n = trimmed.chars().count();
        if n == 0 {
            return String::new();
        }
        if n <= 4 {
            "••••".to_string()
        } else {
            let tail: String = trimmed.chars().skip(n - 4).collect();
            format!("••••{tail}")
        }
    }

    /// Map a row to a [`Provider`] (with the key masked). Shared by readers.
    fn read_provider_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<Provider> {
        Ok(Provider {
            id: row.get(0)?,
            name: row.get(1)?,
            base_url: row.get(2)?,
            // Column 3 is the cleartext key — mask it before exposing.
            api_key_masked: Self::mask_key(&row.get::<_, String>(3)?),
            kind: row.get(4)?,
            created_at: row.get(5)?,
            updated_at: row.get(6)?,
        })
    }

    const PROVIDER_COLUMNS: &'static str =
        "id, name, base_url, api_key, kind, created_at, updated_at";

    /// List all registered providers, alphabetical by name (keys masked).
    pub fn list_providers(&self) -> Result<Vec<Provider>> {
        let sql = format!(
            "SELECT {} FROM providers ORDER BY name ASC",
            Self::PROVIDER_COLUMNS
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], Self::read_provider_row)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Fetch a single provider (key masked).
    pub fn get_provider(&self, id: i64) -> Result<Option<Provider>> {
        let sql = format!("SELECT {} FROM providers WHERE id = ?1", Self::PROVIDER_COLUMNS);
        let row = self
            .conn
            .query_row(&sql, params![id], Self::read_provider_row)
            .optional()?;
        Ok(row)
    }

    /// Fetch a provider's **cleartext** API key (for "Run" / hover-reveal).
    /// Returns `(base_url, api_key)` so the caller can configure the provider.
    pub fn get_provider_endpoint(&self, id: i64) -> Result<Option<(String, String)>> {
        let row = self
            .conn
            .query_row(
                "SELECT base_url, api_key FROM providers WHERE id = ?1",
                params![id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        Ok(row)
    }

    /// Register a new provider. Returns its id.
    pub fn create_provider(&self, name: &str, base_url: &str, api_key: &str, kind: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO providers (name, base_url, api_key, kind)
             VALUES (?1, ?2, ?3, ?4)",
            params![name, base_url, api_key, kind],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update a provider's editable fields.
    pub fn update_provider(
        &self,
        id: i64,
        name: &str,
        base_url: &str,
        api_key: &str,
        kind: &str,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE providers SET name = ?1, base_url = ?2, api_key = ?3, kind = ?4, updated_at = ?5
             WHERE id = ?6",
            params![name, base_url, api_key, kind, now_iso(), id],
        )?;
        Ok(())
    }

    /// Delete a provider. `provider_usage` rows cascade automatically.
    pub fn delete_provider(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM providers WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Record token usage for one turn through a provider.
    pub fn record_provider_usage(&self, provider_id: i64, tokens_in: i64, tokens_out: i64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO provider_usage (provider_id, tokens_in, tokens_out)
             VALUES (?1, ?2, ?3)",
            params![provider_id, tokens_in, tokens_out],
        )?;
        Ok(())
    }

    /// Total tokens (in + out) consumed through a provider in the last hour.
    pub fn provider_usage_last_hour(&self, provider_id: i64) -> Result<i64> {
        // ISO-8601 timestamp for one hour ago.
        let since = (Utc::now() - chrono::Duration::minutes(60)).to_rfc3339();
        let total: i64 = self.conn.query_row(
            "SELECT COALESCE(SUM(tokens_in + tokens_out), 0)
             FROM provider_usage
             WHERE provider_id = ?1 AND ts >= ?2",
            params![provider_id, since],
            |row| row.get(0),
        )?;
        Ok(total)
    }
}

/// Helper: current UTC timestamp in ISO-8601.
fn now_iso() -> String {
    Utc::now().to_rfc3339()
}
