-- Phoenix Agent schema v4 — Science Workbench (Panel 3: tools)
-- Stored inside the SQLCipher-encrypted database.
--
-- Tools are executable scripts the LLM can call like the built-in tools
-- (read_file, run_command, ...). When invoked, the runtime writes the script to
-- a temp file, runs it with the chosen interpreter, feeds the tool arguments as
-- JSON on stdin, and returns stdout. A tool row carries everything needed to
-- build that: the script source, the interpreter, a JSON-Schema describing its
-- parameters, and a read/write classification for the approval gate.
--
-- Per-profile enable/disable uses the same join-table pattern as skills: a
-- profile with NO rows in `profile_tools` inherits `enabled_global = 1`.

CREATE TABLE tools (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL UNIQUE,
    description    TEXT    NOT NULL DEFAULT '',
    interpreter    TEXT    NOT NULL DEFAULT 'python',  -- python|node|sh|powershell|<program>
    script_body    TEXT    NOT NULL DEFAULT '',         -- script source (run from a temp file)
    params_schema  TEXT    NOT NULL DEFAULT '{}',       -- JSON Schema (object) for the parameters
    tool_kind      TEXT    NOT NULL DEFAULT 'write',    -- read | write  (approval gating)
    source         TEXT    NOT NULL DEFAULT 'local',    -- local | github
    source_url     TEXT,                                -- raw URL when installed from github
    enabled_global INTEGER NOT NULL DEFAULT 1,          -- inherited by uncustomized profiles
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Per-profile enable/disable. Absence of a row = inherit enabled_global.
CREATE TABLE profile_tools (
    profile_id  INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    tool_id     INTEGER NOT NULL REFERENCES tools(id)    ON DELETE CASCADE,
    enabled     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (profile_id, tool_id)
);

CREATE INDEX idx_tools_name           ON tools(name);
CREATE INDEX idx_profile_tools_profile ON profile_tools(profile_id);

-- Bump the informational schema-version marker.
INSERT INTO settings (key, value) VALUES ('schema_version', '4')
    ON CONFLICT(key) DO UPDATE SET value = excluded.value;
