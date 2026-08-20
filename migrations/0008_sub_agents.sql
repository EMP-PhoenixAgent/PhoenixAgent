-- Phoenix Agent schema v8 — Sub-Agents (Panel 6)
-- Stored inside the SQLCipher-encrypted database.
--
-- Sub-agents are predefined specialists the main agent can DELEGATE sub-tasks to.
-- Each runs on its configured model (served by AmberCore) with its own persona
-- (system prompt). Stage 1 (this migration) stores them; Stage 2 wires the
-- `delegate` tool + prompt injection so the main agent can actually call them.

CREATE TABLE sub_agents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    description TEXT    NOT NULL DEFAULT '',   -- short specialty, surfaced to the main agent
    persona     TEXT    NOT NULL DEFAULT '',   -- system prompt for the sub-agent
    model       TEXT    NOT NULL DEFAULT '',   -- model tag to run on ('' = active model)
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX idx_sub_agents_name ON sub_agents(name);

-- Bump the informational schema-version marker.
INSERT INTO settings (key, value) VALUES ('schema_version', '8')
    ON CONFLICT(key) DO UPDATE SET value = excluded.value;
