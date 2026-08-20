//! Encrypted SQLite (SQLCipher) connection setup and migrations.

use std::sync::OnceLock;

use rusqlite::Connection;
use rusqlite_migration::{Migrations, M};

use crate::crypto::DerivedKey;
use crate::error::{PhoenixError, Result};

/// The schema migrations, built once.
static MIGRATIONS: OnceLock<Migrations<'static>> = OnceLock::new();

fn migrations() -> &'static Migrations<'static> {
    MIGRATIONS.get_or_init(|| {
        Migrations::new(vec![
            M::up(include_str!("../../migrations/0001_init.sql")),
            M::up(include_str!("../../migrations/0002_workbench.sql")),
            M::up(include_str!("../../migrations/0003_skills.sql")),
            M::up(include_str!("../../migrations/0004_tools.sql")),
            M::up(include_str!("../../migrations/0005_context.sql")),
            M::up(include_str!("../../migrations/0006_memory.sql")),
            M::up(include_str!("../../migrations/0007_providers.sql")),
            M::up(include_str!("../../migrations/0008_sub_agents.sql")),
        ])
    })
}

/// Open the encrypted database, apply the key, and run migrations.
///
/// SQLCipher key is supplied via the `PRAGMA key` hex form. We keep the key in
/// memory only; dropping the [`Connection`] re-encrypts and flushes the file,
/// so the on-disk `.db` is opaque ciphertext while Phoenix is not running.
pub fn open_encrypted(db_path: &std::path::Path, key: &DerivedKey) -> Result<Connection> {
    let mut conn = Connection::open(db_path)?;

    // Supply the raw key in hex form (no passphrase derivation by SQLCipher).
    let hex = key.to_hex();
    let pragma = format!("x'{hex}'");
    conn.pragma_update(None, "key", &pragma)?;

    // Verify the key actually decrypts existing data (no-op on a fresh DB, but
    // catches a wrong passphrase the first time we try to read a table).
    // This select forces SQLCipher to actually decrypt a page.
    let _check: i64 = conn
        .query_row("SELECT count(*) FROM sqlite_master", [], |row| row.get(0))
        .map_err(|e| PhoenixError::Crypto(format!(
            "failed to decrypt database (wrong passphrase or corrupt DB): {e}"
        )))?;

    // Sensible defaults for a local single-user tool.
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;

    // Run any pending migrations.
    migrations()
        .to_latest(&mut conn)
        .map_err(PhoenixError::Migration)?;

    tracing::info!("opened encrypted database at {}", db_path.display());
    Ok(conn)
}
