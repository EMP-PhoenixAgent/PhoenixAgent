//! Portable-layout tests: the one-time migration from the pre-portable home
//! folders (`~/.phoenix` + `~/.ambercore`) into the installation-folder data
//! root, and the models-dir default.

use phoenix_agent::config::{default_models_dir, migrate_legacy_data, save_config, Config, Paths};

fn legacy_home_with_data() -> tempfile::TempDir {
    let home = tempfile::tempdir().expect("tempdir");
    let phoenix = home.path().join(".phoenix");
    std::fs::create_dir_all(&phoenix).unwrap();
    std::fs::write(phoenix.join("memory.db"), b"encrypted-bytes").unwrap();
    std::fs::write(phoenix.join("salt.bin"), b"salt").unwrap();
    // A config that points at the legacy default models dir (the migration
    // must rewrite this to the portable default).
    let mut cfg = Config::default();
    cfg.ambercore_models_dir = Some(
        home.path()
            .join(".ambercore")
            .join("models")
            .to_string_lossy()
            .to_string(),
    );
    save_config(&Paths::new(phoenix.clone()), &cfg).unwrap();

    let models = home.path().join(".ambercore").join("models");
    let sub = models.join("qwen3-8b");
    std::fs::create_dir_all(&sub).unwrap();
    std::fs::write(sub.join("qwen3-8b-q4.gguf"), b"gguf").unwrap();
    std::fs::write(sub.join("qwen3-8b-q4.tokenizer.json"), b"{}").unwrap();
    std::fs::write(models.join("manifest.json"), b"{}").unwrap();
    home
}

#[test]
fn migration_copies_everything_and_rewrites_the_models_dir() {
    let home = legacy_home_with_data();
    let root = tempfile::tempdir().expect("tempdir");
    let paths = Paths::new(root.path().to_path_buf());

    migrate_legacy_data(&paths, home.path());

    // Data files landed in the portable root.
    assert!(paths.db_path.is_file(), "memory.db migrated");
    assert!(paths.salt_path.is_file(), "salt.bin migrated");
    assert!(paths.config_path.is_file(), "config.toml migrated");
    // Models landed in <root>/models, subfolder structure intact.
    let models = default_models_dir(root.path());
    assert!(models.join("qwen3-8b/qwen3-8b-q4.gguf").is_file(), "GGUF migrated");
    assert!(models.join("manifest.json").is_file(), "manifest migrated");
    // The migrated config no longer points at the legacy dir.
    let cfg = phoenix_agent::config::load_config(&paths).unwrap();
    assert_eq!(cfg.ambercore_models_dir, None, "legacy models dir reset");
    // Legacy data is untouched (copied, never deleted).
    assert!(home.path().join(".phoenix/memory.db").is_file());
    assert!(home.path().join(".ambercore/models/manifest.json").is_file());
}

#[test]
fn migration_is_noop_when_the_portable_root_is_already_initialized() {
    let home = legacy_home_with_data();
    let root = tempfile::tempdir().expect("tempdir");
    let paths = Paths::new(root.path().to_path_buf());
    // An existing config means this is a fresh install (or already migrated).
    std::fs::write(&paths.config_path, "").unwrap();

    migrate_legacy_data(&paths, home.path());

    assert!(!paths.db_path.exists(), "nothing copied over the fresh config");
    assert!(!default_models_dir(root.path()).join("manifest.json").exists());
}

#[test]
fn migration_never_overwrites_existing_destination_files() {
    let home = legacy_home_with_data();
    let root = tempfile::tempdir().expect("tempdir");
    let paths = Paths::new(root.path().to_path_buf());
    std::fs::create_dir_all(&paths.data_dir).unwrap();
    std::fs::write(&paths.db_path, b"existing-newer-db").unwrap();

    migrate_legacy_data(&paths, home.path());

    let bytes = std::fs::read(&paths.db_path).unwrap();
    assert_eq!(bytes, b"existing-newer-db", "destination file preserved");
}

#[test]
fn migration_without_legacy_dirs_touches_nothing() {
    let home = tempfile::tempdir().expect("tempdir"); // no .phoenix / .ambercore
    let root = tempfile::tempdir().expect("tempdir");
    let paths = Paths::new(root.path().to_path_buf());

    migrate_legacy_data(&paths, home.path());

    assert!(!paths.config_path.exists(), "no config invented");
}

#[test]
fn portable_models_dir_lives_inside_the_data_root() {
    let root = std::path::Path::new("C:/Apps/Phoenix Agent");
    assert_eq!(default_models_dir(root), root.join("models"));
}
