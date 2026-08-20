-- Phoenix Agent schema v2 — Science Workbench (Panel 1: profiles)
-- Stored inside the SQLCipher-encrypted database.
--
-- A profile bundles agent *behavior* settings so the user can switch between
-- presets (e.g. a cautious "review" profile vs. an "aggressive" one). The
-- active model and working directory are NOT profile fields — they live in the
-- `settings` KV table so they can be switched independently from the sidebar.
--
-- Future panels (skills/tools/context/memory) will add their own master tables
-- and `profile_*` join tables in later migrations; only `profiles` is created
-- here to avoid unused schema.

CREATE TABLE profiles (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT    NOT NULL UNIQUE,
    approval_policy  TEXT    NOT NULL DEFAULT 'writes_only',  -- all | writes_only | never
    max_iterations   INTEGER NOT NULL DEFAULT 25,
    context_window   INTEGER NOT NULL DEFAULT 50,
    is_default       INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX idx_profiles_default ON profiles(is_default);

-- Bump the informational schema-version marker.
INSERT INTO settings (key, value) VALUES ('schema_version', '2')
    ON CONFLICT(key) DO UPDATE SET value = excluded.value;
