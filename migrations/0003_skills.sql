-- Phoenix Agent schema v3 — Science Workbench (Panel 2: skills)
-- Stored inside the SQLCipher-encrypted database.
--
-- Skills are markdown knowledge files injected into the agent's system prompt.
-- A skill is either authored locally (`source = 'local'`) or installed from a
-- GitHub raw file (`source = 'github'`, `source_url` = raw URL).
--
-- Per-profile enable/disable is modeled with a `profile_skills` join table. A
-- profile with NO rows in `profile_skills` inherits the global default
-- (`enabled_global = 1`) skills; once the user customizes a profile, the join
-- table becomes the source of truth for that profile (handled in Rust).

CREATE TABLE skills (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL UNIQUE,
    description    TEXT    NOT NULL DEFAULT '',
    body           TEXT    NOT NULL DEFAULT '',        -- markdown injected into the prompt
    source         TEXT    NOT NULL DEFAULT 'local',   -- local | github
    source_url     TEXT,                                -- raw.githubusercontent.com URL if github
    enabled_global INTEGER NOT NULL DEFAULT 1,         -- included in profiles that haven't customized
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- Per-profile enable/disable. Absence of a row = "inherit enabled_global".
CREATE TABLE profile_skills (
    profile_id  INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    skill_id    INTEGER NOT NULL REFERENCES skills(id)   ON DELETE CASCADE,
    enabled     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (profile_id, skill_id)
);

CREATE INDEX idx_skills_name           ON skills(name);
CREATE INDEX idx_profile_skills_profile ON profile_skills(profile_id);

-- Bump the informational schema-version marker.
INSERT INTO settings (key, value) VALUES ('schema_version', '3')
    ON CONFLICT(key) DO UPDATE SET value = excluded.value;
