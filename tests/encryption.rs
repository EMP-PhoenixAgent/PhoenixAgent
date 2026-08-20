//! Integration test: verifies the full encryption pipeline
//! (passphrase → Argon2 → SQLCipher unlock → migration → round-trip).
//!
//! Run with: cargo test --test encryption

use phoenix_agent::crypto::{
    derive_key, derive_wrap_key, load_or_create_salt, rotate_salt, unwrap_key, wrap_key, DerivedKey,
    KeyBundle,
};
use phoenix_agent::crypto::totp as totp_lib;
use phoenix_agent::db::{open_encrypted, MemoryStore};
use phoenix_agent::model::{ChatMessage, ChatRole};

#[test]
fn encrypted_db_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("test.db");
    let salt_path = tmp.path().join("salt.bin");

    let passphrase = "correct horse battery staple";
    let salt = load_or_create_salt(&salt_path).expect("salt");
    let key = derive_key(passphrase, &salt, None).expect("derive key");

    // First open: creates the DB + runs migrations.
    {
        let conn = open_encrypted(&db_path, &key).expect("open (create)");
        let store = MemoryStore::new(conn);
        let sid = store
            .start_session(None, "test-model", "round-trip test")
            .expect("start session");
        let msg = ChatMessage::user("hello, encrypted world");
        store
            .append_message(sid, &msg, Some("test-model"), Some(10), Some(5))
            .expect("append");
        let loaded = store.load_messages(sid, 100).expect("load");
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].content, "hello, encrypted world");
        assert_eq!(loaded[0].role, ChatRole::User);
    }
    // Connection dropped → DB is re-encrypted and flushed to disk.

    // The DB file must exist and not be plain text.
    let raw = std::fs::read(&db_path).expect("read db file");
    assert!(raw.len() > 100, "db file should have content");
    let as_text = String::from_utf8_lossy(&raw);
    assert!(
        !as_text.contains("round-trip test"),
        "plaintext must NOT appear in encrypted DB file"
    );

    // Reopen with the SAME key: data should be there.
    {
        let conn = open_encrypted(&db_path, &key).expect("open (reopen)");
        let store = MemoryStore::new(conn);
        let sessions = store.list_sessions().expect("list");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].title, "round-trip test");
    }

    // Reopen with the WRONG key: must fail.
    {
        let bad_key = derive_key("wrong passphrase entirely", &salt, None).expect("derive bad key");
        let result = open_encrypted(&db_path, &bad_key);
        assert!(
            result.is_err(),
            "wrong passphrase must NOT unlock the database"
        );
    }
}

/// Profiles: ensure_default_profile creates a default; active-profile and
/// workdir settings round-trip through the encrypted `settings` table.
#[test]
fn profiles_and_settings_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("profiles.db");
    let salt_path = tmp.path().join("salt.bin");

    let salt = load_or_create_salt(&salt_path).expect("salt");
    let key = derive_key("correct horse battery staple", &salt, None).expect("derive key");

    let conn = open_encrypted(&db_path, &key).expect("open");
    let store = MemoryStore::new(conn);

    // Empty DB → ensure_default_profile seeds exactly one default profile.
    let default_id = store.ensure_default_profile().expect("ensure default");
    assert!(default_id > 0);
    let profiles = store.list_profiles().expect("list profiles");
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].name, "Default");
    assert!(profiles[0].is_default);

    // Idempotent: a second call returns the same id, no duplicate row.
    let again = store.ensure_default_profile().expect("ensure default again");
    assert_eq!(again, default_id);
    assert_eq!(store.list_profiles().unwrap().len(), 1);

    // Create a second profile and switch active to it.
    let custom_id = store
        .create_profile("Cautious", "all", 10, 30)
        .expect("create profile");
    assert_ne!(custom_id, default_id);
    store.set_active_profile_id(custom_id).expect("set active");
    let active = store.get_active_profile_id().expect("get active");
    assert_eq!(active, Some(custom_id));

    // Workdir persists through the settings KV table.
    store.set_workdir("N:\\Some Project").expect("set workdir");
    assert_eq!(
        store.get_workdir().expect("get workdir"),
        Some("N:\\Some Project".to_string())
    );
}

/// Skills: seed, per-profile enable/disable, profile inheritance, cascade delete.
#[test]
fn skills_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("skills.db");
    let salt_path = tmp.path().join("salt.bin");

    let salt = load_or_create_salt(&salt_path).expect("salt");
    let key = derive_key("correct horse battery staple", &salt, None).expect("derive key");

    let conn = open_encrypted(&db_path, &key).expect("open");
    let store = MemoryStore::new(conn);

    // Need a profile to attach skill enablement to.
    let pid = store.ensure_default_profile().expect("ensure default");

    // Seeding is idempotent and creates the starter skills.
    store.ensure_seed_skills().expect("seed once");
    let seeded = store.list_skills().expect("list");
    assert!(!seeded.is_empty(), "starter skills should be seeded");
    store.ensure_seed_skills().expect("seed again (idempotent)");
    assert_eq!(
        store.list_skills().unwrap().len(),
        seeded.len(),
        "second seed should not duplicate"
    );

    // A profile with no explicit profile_skills rows inherits enabled_global=1.
    let inherited = store
        .list_enabled_skills_for_profile(pid)
        .expect("enabled (inherited)");
    assert_eq!(inherited.len(), seeded.len());

    // Create two custom skills (one enabled_global, one disabled).
    let on_id = store
        .create_skill("alpha", "a skill", "body-a", "local", None)
        .expect("create on");
    let off_id = store
        .create_skill("beta", "b skill", "body-b", "local", None)
        .expect("create off");
    store
        .set_skill_enabled_for_profile(pid, off_id, false)
        .expect("disable beta");
    // Once we customize, the inheritance stops: only explicitly-enabled skills
    // apply (alpha + the seeded ones are NOT auto-included anymore because the
    // profile now has profile_skills rows — but only off_id got a row, enabled=0).
    let customized = store
        .list_enabled_skills_for_profile(pid)
        .expect("enabled (customized)");
    assert!(
        customized.iter().all(|s| s.id != off_id),
        "disabled skill must not appear"
    );

    // Enable alpha for the profile.
    store
        .set_skill_enabled_for_profile(pid, on_id, true)
        .expect("enable alpha");
    let with_alpha = store
        .list_enabled_skills_for_profile(pid)
        .expect("enabled (with alpha)");
    assert!(with_alpha.iter().any(|s| s.id == on_id));

    // list_skills_for_profile reports enabled state for every skill.
    let panel = store.list_skills_for_profile(pid).expect("panel list");
    let alpha_row = panel.iter().find(|r| r.skill.id == on_id).unwrap();
    assert!(alpha_row.enabled);
    let beta_row = panel.iter().find(|r| r.skill.id == off_id).unwrap();
    assert!(!beta_row.enabled);

    // Update a skill's body.
    store
        .update_skill(on_id, "alpha", "updated", "new-body")
        .expect("update");
    let updated = store.get_skill(on_id).unwrap().unwrap();
    assert_eq!(updated.body, "new-body");
    assert_eq!(updated.description, "updated");

    // Delete cascades to profile_skills.
    store.delete_skill(off_id).expect("delete");
    assert!(store.get_skill(off_id).unwrap().is_none());
    let remaining = store.list_skills().unwrap();
    assert!(remaining.iter().all(|s| s.id != off_id));
}

/// Passphrase change via SQLCipher PRAGMA hexrekey: data survives a key change
/// performed in place. Mirrors what `change_passphrase` does.
#[test]
fn rekey_preserves_data() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("rekey.db");
    let salt_path = tmp.path().join("salt.bin");

    let salt = load_or_create_salt(&salt_path).expect("salt");
    let old_key = derive_key("old-passphrase", &salt, None).expect("old key");

    // Create + write a row with the old key.
    {
        let conn = open_encrypted(&db_path, &old_key).expect("open old");
        let store = MemoryStore::new(conn);
        let sid = store
            .start_session(None, "m", "survive-rekey")
            .expect("start");
        store
            .append_message(sid, &ChatMessage::user("keep me"), None, None, None)
            .expect("append");
    }

    // Rekey in place: rotate salt, derive new key, rekey.
    let new_salt = rotate_salt(&salt_path).expect("rotate salt");
    let new_key = derive_key("new-passphrase", &new_salt, None).expect("new key");
    {
        let conn = open_encrypted(&db_path, &old_key).expect("reopen with old key to rekey");
        // Checkpoint WAL so all pages are written before rekeying.
        let _ = conn.pragma_update(None, "wal_checkpoint", "TRUNCATE");
        // SQLCipher raw-key rekey. We opened with the x'<hex>' raw form, so we
        // rekey with the same raw form via the `rekey` pragma wrapping the hex
        // in x'...'. (hexrekey with bare hex was unreliable across builds.)
        let new_pragma = format!("x'{}'", new_key.to_hex());
        conn.pragma_update(None, "rekey", new_pragma)
            .expect("rekey");
        // Drop the connection to flush.
    }

    // Old key must now fail.
    {
        let result = open_encrypted(&db_path, &old_key);
        assert!(result.is_err(), "old key must not work after rekey");
    }

    // New key works and the data is intact.
    {
        let conn = open_encrypted(&db_path, &new_key).expect("open with new key");
        let store = MemoryStore::new(conn);
        let sessions = store.list_sessions().expect("list");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].title, "survive-rekey");
    }
}

/// TOTP 2FA: a code derived key differs from the no-code key (so 2FA actually
/// changes the derived key), and a valid current code verifies.
#[test]
fn totp_changes_key_and_verifies() {
    let salt = load_or_create_salt(&std::env::temp_dir().join("phx_totp_salt.bin"))
        .expect("salt");

    // Same passphrase, with vs without a code → different keys.
    let k_no_code = derive_key("hunter2hunter2", &salt, None).expect("no-code key");
    let k_with_code = derive_key("hunter2hunter2", &salt, Some("123456")).expect("with-code key");
    assert_ne!(
        k_no_code.to_hex(),
        k_with_code.to_hex(),
        "a 2FA code must change the derived key"
    );

    // Generate a TOTP and verify a fresh code round-trips.
    let instance = totp_lib::generate("user@example.com").expect("generate");
    let code = totp_lib::current_code(&instance.totp);
    assert_eq!(code.len(), 6);
    assert!(
        totp_lib::verify(&instance.totp, &code),
        "a freshly generated code must verify"
    );
    assert!(
        !totp_lib::verify(&instance.totp, "000000") || code == "000000",
        "a wrong code should not verify (unless it happened to equal the real one)"
    );

    // from_secret reconstructs the same TOTP (same current code).
    let rebuilt = totp_lib::from_secret(&instance.secret_b32, "user@example.com")
        .expect("from_secret");
    assert_eq!(totp_lib::current_code(&rebuilt), code);

    // validate_secret accepts the generated base32.
    assert!(totp_lib::validate_secret(&instance.secret_b32).is_ok());
}

/// Key wrapping: wrap/unwrap round-trips, a tampered blob fails cleanly, and a
/// wrong wrap key fails — never silently yielding a corrupted key.
#[test]
fn key_wrap_round_trip() {
    let salt = phoenix_agent::crypto::random_salt();
    let plain = derive_key("the real db password", &salt, None).expect("derive key");

    // Correct wrap key → round-trip succeeds.
    let wk = derive_wrap_key("launch-password-1", &salt).expect("wrap key");
    let blob = wrap_key(&plain, &wk).expect("wrap");
    let unwrapped = unwrap_key(&blob, &wk).expect("unwrap");
    assert_eq!(plain.to_hex(), unwrapped.to_hex(), "unwrap must return the wrapped key");

    // Wrong wrap key → error, not garbage.
    let wrong_wk = derive_wrap_key("launch-password-2", &salt).expect("wrap key");
    assert!(unwrap_key(&blob, &wrong_wk).is_err(), "a wrong wrap key must fail");

    // Tampered blob → error (GCM tag mismatch), not a silent corruption.
    let mut tampered = blob.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xFF;
    assert!(unwrap_key(&tampered, &wk).is_err(), "a tampered blob must fail");
}

/// The full key-bundle flow: create a DB, derive its key, wrap it under a
/// launch password, then unwrap via the bundle to reopen the same DB — proving
/// the wrapped path opens exactly the DB the agent created.
#[test]
fn launch_unlock_uses_wrapped_key() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("wrapped.db");
    let bundle_path = tmp.path().join("keys.phx");
    let salt_path = tmp.path().join("salt.bin");

    // The DB key is derived from the (default) DB password once.
    let salt = load_or_create_salt(&salt_path).expect("salt");
    let db_key = derive_key("PhoenixAgent", &salt, None).expect("db key");
    {
        let conn = open_encrypted(&db_path, &db_key).expect("create db");
        let store = MemoryStore::new(conn);
        store.ensure_default_profile().expect("default profile");
    } // DB created + closed.

    // Wrap the DB key under the launch password and persist the bundle.
    let bundle = KeyBundle::create(&db_key, "user-launch-pw").expect("create bundle");
    bundle.save(&bundle_path).expect("save bundle");

    // Reload the bundle and unwrap with the launch password → must open the DB.
    let loaded = KeyBundle::load(&bundle_path).expect("load bundle");
    let recovered = loaded.unwrap_primary("user-launch-pw").expect("unwrap");
    assert_eq!(recovered.to_hex(), db_key.to_hex(), "bundle must unwrap the same key");

    let conn = open_encrypted(&db_path, &recovered).expect("reopen with unwrapped key");
    let store = MemoryStore::new(conn);
    assert!(store.get_active_profile_id().expect("profiles").is_some(),
        "the unwrapped key must open the same DB the agent created");

    // A wrong launch password fails cleanly.
    assert!(loaded.unwrap_primary("wrong-pw").is_err());
}

/// Changing the launch password only re-wraps — the DB key (and thus the DB)
/// is untouched. A DB password change, by contrast, produces a different key.
#[test]
fn change_launch_rewraps_not_rekeys() {
    let salt = phoenix_agent::crypto::random_salt();
    let db_key: DerivedKey =
        derive_key("PhoenixAgent", &salt, None).expect("db key");

    let mut bundle = KeyBundle::create(&db_key, "launch-old").expect("bundle");
    // Change the launch password.
    bundle.change_primary(&db_key, "launch-new").expect("change");
    // Old password no longer works.
    assert!(bundle.unwrap_primary("launch-old").is_err());
    // New password unwraps the SAME db key — no rekey happened.
    let recovered = bundle.unwrap_primary("launch-new").expect("unwrap new");
    assert_eq!(recovered.to_hex(), db_key.to_hex(),
        "changing the launch password must NOT change the DB key");
}

/// Tools: create, per-profile enable/disable, inheritance, cascade delete.
#[test]
fn tools_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("tools.db");
    let salt_path = tmp.path().join("salt.bin");

    let salt = load_or_create_salt(&salt_path).expect("salt");
    let key = derive_key("correct horse battery staple", &salt, None).expect("derive key");
    let conn = open_encrypted(&db_path, &key).expect("open");
    let store = MemoryStore::new(conn);

    let pid = store.ensure_default_profile().expect("ensure default");

    // No tools seeded — list is empty.
    assert!(store.list_tools().unwrap().is_empty());

    // Create a read tool + a write tool.
    let read_id = store
        .create_tool("word_count", "count words", "sh", "wc", "{}", "read", "local", None)
        .expect("create read tool");
    let write_id = store
        .create_tool("make_dir", "mkdir", "sh", "mkdir", "{}", "write", "local", None)
        .expect("create write tool");

    // A profile with no profile_tools rows inherits enabled_global=1.
    let inherited = store.list_enabled_tools_for_profile(pid).unwrap();
    assert_eq!(inherited.len(), 2);

    // Customizing: explicitly enable read_id, disable write_id. Once a profile
    // has ANY profile_tools row it stops inheriting — only explicitly-enabled
    // tools appear (a tool with no row is treated as disabled).
    store.set_tool_enabled_for_profile(pid, read_id, true).unwrap();
    store.set_tool_enabled_for_profile(pid, write_id, false).unwrap();
    let customized = store.list_enabled_tools_for_profile(pid).unwrap();
    assert_eq!(customized.len(), 1);
    assert_eq!(customized[0].name, "word_count");

    // Panel view reports enabled state for every tool.
    let panel = store.list_tools_for_profile(pid).unwrap();
    let read_row = panel.iter().find(|r| r.tool.id == read_id).unwrap();
    assert!(read_row.enabled);
    let write_row = panel.iter().find(|r| r.tool.id == write_id).unwrap();
    assert!(!write_row.enabled);

    // Update a tool.
    store.update_tool(read_id, "word_count", "updated desc", "sh", "wc -w", "{}", "read").unwrap();
    let updated = store.get_tool(read_id).unwrap().unwrap();
    assert_eq!(updated.description, "updated desc");

    // Delete cascades to profile_tools.
    store.delete_tool(write_id).unwrap();
    assert!(store.get_tool(write_id).unwrap().is_none());
    let remaining = store.list_tools().unwrap();
    assert!(remaining.iter().all(|t| t.id != write_id));
}

/// A user script tool actually executes: a trivial `sh` script echoes the JSON
/// it read on stdin, proving the interpreter+stdin path works end-to-end.
#[tokio::test]
async fn user_script_executes() {
    use phoenix_agent::agent::tools::user_script::UserScriptTool;
    use phoenix_agent::agent::tools::{Tool, ToolContext};
    use phoenix_agent::config::ToolKind;
    use serde_json::json;
    use std::path::PathBuf;

    // `cat` echoes stdin straight back; on a real tool the script would parse it.
    let tool = UserScriptTool::new(
        "echo_args",
        "echo the JSON args back",
        "sh",
        "cat",
        json!({}),
        ToolKind::Read,
    );
    let ctx = ToolContext {
        workdir: PathBuf::from("."),
        os: std::env::consts::OS.to_string(),
    };
    let args = json!({ "hello": "world" });
    let result = tool.run(&args, &ctx).await;
    assert!(
        result.success,
        "echo_args should succeed; content was: {}",
        result.content
    );
    // The script received the args JSON we sent on stdin.
    assert!(
        result.content.contains("hello") && result.content.contains("world"),
        "stdout should contain the echoed JSON; got: {}",
        result.content
    );
}

/// Context files: create, per-profile enable/disable, inheritance, cascade delete.
#[test]
fn context_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("context.db");
    let salt_path = tmp.path().join("salt.bin");

    let salt = load_or_create_salt(&salt_path).expect("salt");
    let key = derive_key("correct horse battery staple", &salt, None).expect("derive key");
    let conn = open_encrypted(&db_path, &key).expect("open");
    let store = MemoryStore::new(conn);

    let pid = store.ensure_default_profile().expect("ensure default");

    // No context seeded — list is empty.
    assert!(store.list_context().unwrap().is_empty());

    // Create two context files.
    let db_id = store
        .create_context("database", "which DB we use", "We use PostgreSQL, not MySQL.")
        .expect("create db context");
    let tests_id = store
        .create_context("test-runner", "how to run tests", "Run tests with `pnpm test`.")
        .expect("create tests context");

    // A profile with no profile_context rows inherits enabled_global=1.
    let inherited = store.list_enabled_context_for_profile(pid).unwrap();
    assert_eq!(inherited.len(), 2);

    // Customizing: enable one, disable the other.
    store.set_context_enabled_for_profile(pid, db_id, true).unwrap();
    store.set_context_enabled_for_profile(pid, tests_id, false).unwrap();
    let customized = store.list_enabled_context_for_profile(pid).unwrap();
    assert_eq!(customized.len(), 1);
    assert_eq!(customized[0].name, "database");

    // Panel view reports enabled state for every file.
    let panel = store.list_context_for_profile(pid).unwrap();
    let db_row = panel.iter().find(|r| r.context.id == db_id).unwrap();
    assert!(db_row.enabled);
    let tests_row = panel.iter().find(|r| r.context.id == tests_id).unwrap();
    assert!(!tests_row.enabled);

    // Update a context file.
    store.update_context(db_id, "database", "updated", "We use PostgreSQL 16.").unwrap();
    let updated = store.get_context(db_id).unwrap().unwrap();
    assert_eq!(updated.body, "We use PostgreSQL 16.");

    // Delete cascades to profile_context.
    store.delete_context(tests_id).unwrap();
    assert!(store.get_context(tests_id).unwrap().is_none());
    let remaining = store.list_context().unwrap();
    assert!(remaining.iter().all(|c| c.id != tests_id));
}

/// Memory sources (MCP connections): create, per-profile enable/disable,
/// inheritance, cascade delete — mirrors the context round-trip.
#[test]
fn memory_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("memory.db");
    let salt_path = tmp.path().join("salt.bin");

    let salt = load_or_create_salt(&salt_path).expect("salt");
    let key = derive_key("correct horse battery staple", &salt, None).expect("derive key");
    let conn = open_encrypted(&db_path, &key).expect("open");
    let store = MemoryStore::new(conn);

    let pid = store.ensure_default_profile().expect("ensure default");

    // No memory seeded — list is empty.
    assert!(store.list_memory().unwrap().is_empty());

    // Create two MCP connections (one stdio, one http).
    let fs_id = store
        .create_memory(
            "filesystem",
            "local fs server",
            "stdio",
            "npx",
            r#"["@modelcontextprotocol/server-filesystem", "/tmp"]"#,
        )
        .expect("create fs memory");
    let http_id = store
        .create_memory(
            "remote",
            "remote http server",
            "http",
            "https://example.com/mcp",
            "[]",
        )
        .expect("create http memory");

    // A profile with no profile_memory rows inherits enabled_global=1.
    let inherited = store.list_enabled_memory_for_profile(pid).unwrap();
    assert_eq!(inherited.len(), 2);

    // Customizing: enable one, disable the other.
    store.set_memory_enabled_for_profile(pid, fs_id, true).unwrap();
    store.set_memory_enabled_for_profile(pid, http_id, false).unwrap();
    let customized = store.list_enabled_memory_for_profile(pid).unwrap();
    assert_eq!(customized.len(), 1);
    assert_eq!(customized[0].name, "filesystem");
    assert_eq!(customized[0].transport, "stdio");

    // Panel view reports enabled state for every source.
    let panel = store.list_memory_for_profile(pid).unwrap();
    let fs_row = panel.iter().find(|r| r.source.id == fs_id).unwrap();
    assert!(fs_row.enabled);
    let http_row = panel.iter().find(|r| r.source.id == http_id).unwrap();
    assert!(!http_row.enabled);

    // Update a memory source.
    store
        .update_memory(
            http_id,
            "remote",
            "updated desc",
            "http",
            "https://example.com/mcp/v2",
            "[]",
        )
        .unwrap();
    let updated = store.get_memory(http_id).unwrap().unwrap();
    assert_eq!(updated.command, "https://example.com/mcp/v2");
    assert_eq!(updated.description, "updated desc");

    // Delete cascades to profile_memory.
    store.delete_memory(fs_id).unwrap();
    assert!(store.get_memory(fs_id).unwrap().is_none());
    let remaining = store.list_memory().unwrap();
    assert!(remaining.iter().all(|m| m.id != fs_id));
}

/// A bogus stdio connection must return a clean error, not panic. This exercises
/// the MCP transport's failure path without needing a live server fixture.
#[tokio::test]
async fn mcp_connect_failure_returns_error() {
    use phoenix_agent::agent::mcp::{McpClient, McpTransport};

    let transport = McpTransport::Stdio {
        // A command that does not exist on PATH.
        command: "phoenix-definitely-not-a-real-mcp-server-xyz".into(),
        args: vec![],
    };
    let result = McpClient::connect(transport).await;
    assert!(result.is_err(), "connecting to a nonexistent server should error");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("spawn") || msg.contains("MCP"),
        "error should mention the spawn failure, got: {msg}"
    );
}


