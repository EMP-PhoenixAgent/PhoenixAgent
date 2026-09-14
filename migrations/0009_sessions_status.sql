-- Session learning/compaction status — powers the session-history dash
-- colors: green = learned + compacted, orange = learned only, red = wild
-- (never learned). NULL = action never taken for that session.
ALTER TABLE sessions ADD COLUMN learned_at TEXT;
ALTER TABLE sessions ADD COLUMN compacted_at TEXT;
