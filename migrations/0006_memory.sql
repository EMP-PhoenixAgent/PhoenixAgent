-- Phoenix Agent schema v6 — Science Workbench (Panel 5: memory / MCP)
-- Stored inside the SQLCipher-encrypted database.
--
-- A "memory source" is an external MCP (Model Context Protocol) server
-- connection. Each enabled connection exposes its tools to the agent (they are
-- registered alongside the built-in and user-script tools) and may expose
-- resources. This is the most architecturally distinct panel: unlike
-- skills/tools/context (which are locally-authored markdown/scripts injected
-- into the prompt), memory sources are LIVE external connections spawned or
-- contacted at reload time.
--
-- Two transports are supported (see `transport` column):
--   - "stdio": spawn `command` (with JSON args) and speak JSON-RPC 2.0 over its
--     stdin/stdout (newline-delimited). `command` is the executable; `args_json`
--     is a JSON array of string args.
--   - "http":  POST JSON-RPC 2.0 requests to the URL in `command`.
--
-- Per-profile enable/disable uses the same join-table pattern as skills/tools/
-- context: a profile with NO rows in `profile_memory` inherits `enabled_global`.

CREATE TABLE memory_sources (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL UNIQUE,
    description    TEXT    NOT NULL DEFAULT '',
    transport      TEXT    NOT NULL DEFAULT 'stdio',   -- 'stdio' | 'http'
    command        TEXT    NOT NULL DEFAULT '',         -- executable (stdio) or base URL (http)
    args_json      TEXT    NOT NULL DEFAULT '[]',       -- JSON array of string args (stdio) / extra config
    enabled_global INTEGER NOT NULL DEFAULT 1,          -- inherited by uncustomized profiles
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Per-profile enable/disable. Absence of a row = inherit enabled_global.
CREATE TABLE profile_memory (
    profile_id INTEGER NOT NULL REFERENCES profiles(id)      ON DELETE CASCADE,
    memory_id  INTEGER NOT NULL REFERENCES memory_sources(id) ON DELETE CASCADE,
    enabled    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (profile_id, memory_id)
);

CREATE INDEX idx_memory_sources_name     ON memory_sources(name);
CREATE INDEX idx_profile_memory_profile  ON profile_memory(profile_id);

-- Bump the informational schema-version marker.
INSERT INTO settings (key, value) VALUES ('schema_version', '6')
    ON CONFLICT(key) DO UPDATE SET value = excluded.value;
