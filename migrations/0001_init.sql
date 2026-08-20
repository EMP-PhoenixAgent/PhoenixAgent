-- Phoenix Agent schema v1
-- Stored inside a SQLCipher-encrypted SQLite database.

CREATE TABLE projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    path        TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    title       TEXT    NOT NULL DEFAULT 'Untitled',
    model       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role         TEXT    NOT NULL,              -- system | user | assistant | tool
    content      TEXT    NOT NULL DEFAULT '',
    tool_calls   TEXT,                           -- JSON array (assistant tool-call requests)
    tool_name    TEXT,                           -- name of tool (role=tool rows)
    model        TEXT,
    tokens_in    INTEGER,
    tokens_out   INTEGER,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  INTEGER REFERENCES projects(id) ON DELETE SET NULL,
    kind        TEXT    NOT NULL DEFAULT 'note',  -- note | fact | pin
    content     TEXT    NOT NULL,
    pinned      INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE INDEX idx_messages_session ON messages(session_id, created_at);
CREATE INDEX idx_sessions_project ON sessions(project_id);
CREATE INDEX idx_notes_project    ON notes(project_id);

-- Track schema version for future migrations
INSERT INTO settings (key, value) VALUES ('schema_version', '1');
