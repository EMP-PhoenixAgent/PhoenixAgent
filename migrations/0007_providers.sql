-- Phoenix Agent schema v7 — Models panel v0.5: cloud provider registry
-- Stored inside the SQLCipher-encrypted database.
--
-- A "provider" is a cloud OpenAI-compatible API endpoint the user registers
-- from the Models panel's red box (e.g. OpenAI, OpenRouter, Together, Groq, or
-- a local OpenAI-compatible server). The API key lives here, encrypted at rest
-- by SQLCipher (AES-256) — it is never written to plaintext config.
--
-- `provider_usage` records per-turn token consumption so the panel can show
-- "tokens used in the last hour" per provider. One row per assistant turn made
-- through that provider. Rows older than the window can be pruned by a future
-- maintenance task; the panel queries a 1-hour sliding window.

CREATE TABLE providers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,                 -- user-chosen label
    base_url    TEXT    NOT NULL,                         -- e.g. https://api.openai.com
    api_key     TEXT    NOT NULL,                         -- bearer secret (encrypted at rest)
    kind        TEXT    NOT NULL DEFAULT 'openai',        -- protocol family; 'openai' for v1
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

-- One row per model turn made through a provider. Used for the "last hour"
-- consumption figure in the Provider API list.
CREATE TABLE provider_usage (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id  INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    tokens_in    INTEGER NOT NULL DEFAULT 0,
    tokens_out   INTEGER NOT NULL DEFAULT 0,
    ts           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX idx_providers_name        ON providers(name);
CREATE INDEX idx_provider_usage_p_id   ON provider_usage(provider_id);
CREATE INDEX idx_provider_usage_ts     ON provider_usage(ts);

-- Bump the informational schema-version marker.
INSERT INTO settings (key, value) VALUES ('schema_version', '7')
    ON CONFLICT(key) DO UPDATE SET value = excluded.value;
