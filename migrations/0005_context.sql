-- Phoenix Agent schema v5 — Science Workbench (Panel 4: context)
-- Stored inside the SQLCipher-encrypted database.
--
-- Context files are declarative facts about the current project (e.g. "we use
-- PostgreSQL", "auth lives in src/auth/"). They are injected into the system
-- prompt like skills, but under a distinct "## Project Context" section framed
-- as authoritative ground truth the model must respect — not optional
-- methodology. Context is always locally authored (no GitHub install).
--
-- Per-profile enable/disable uses the same join-table pattern as skills/tools:
-- a profile with NO rows in `profile_context` inherits `enabled_global = 1`.

CREATE TABLE context_files (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL UNIQUE,
    description    TEXT    NOT NULL DEFAULT '',
    body           TEXT    NOT NULL DEFAULT '',        -- markdown injected as ground truth
    enabled_global INTEGER NOT NULL DEFAULT 1,          -- inherited by uncustomized profiles
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Per-profile enable/disable. Absence of a row = inherit enabled_global.
CREATE TABLE profile_context (
    profile_id  INTEGER NOT NULL REFERENCES profiles(id)     ON DELETE CASCADE,
    context_id  INTEGER NOT NULL REFERENCES context_files(id) ON DELETE CASCADE,
    enabled     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (profile_id, context_id)
);

CREATE INDEX idx_context_files_name     ON context_files(name);
CREATE INDEX idx_profile_context_profile ON profile_context(profile_id);

-- Bump the informational schema-version marker.
INSERT INTO settings (key, value) VALUES ('schema_version', '5')
    ON CONFLICT(key) DO UPDATE SET value = excluded.value;
