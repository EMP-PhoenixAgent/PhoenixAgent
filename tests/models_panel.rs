//! Integration tests for the Models panel v0.5 backend:
//! - The provider registry (encrypted DB CRUD + usage windowing)
//! - The dispatch provider's route switching (local ↔ cloud)
//!
//! Run with: cargo test --test models_panel

use phoenix_agent::crypto::{derive_key, load_or_create_salt};
use phoenix_agent::db::{open_encrypted, MemoryStore};
use phoenix_agent::model::dispatch::{ActiveRoute, CloudRoute, DispatchProvider, LocalBackend};
use phoenix_agent::model::ModelProvider;

/// Open a fresh encrypted DB at a temp path with all migrations applied.
fn fresh_store() -> (tempfile::TempDir, MemoryStore) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let salt = load_or_create_salt(&tmp.path().join("salt.bin")).expect("salt");
    let key = derive_key("test-passphrase", &salt, None).expect("derive key");
    let conn = open_encrypted(&tmp.path().join("test.db"), &key).expect("open");
    let store = MemoryStore::new(conn);
    (tmp, store)
}

#[test]
fn provider_crud_and_masking() {
    let (_tmp, store) = fresh_store();

    // Create two providers.
    let id1 = store
        .create_provider("OpenAI", "https://api.openai.com", "sk-abcd1234", "openai")
        .expect("create");
    let id2 = store
        .create_provider("OpenRouter", "https://openrouter.ai", "or-xyz9876", "openai")
        .expect("create");

    // List returns them with MASKED keys (never cleartext).
    let list = store.list_providers().expect("list");
    assert_eq!(list.len(), 2);
    let openai = list.iter().find(|p| p.name == "OpenAI").unwrap();
    assert_eq!(openai.api_key_masked, "••••1234"); // last 4 chars
    assert!(!openai.api_key_masked.contains("sk-abcd1234")); // never the full key

    // The cleartext key is only available via the explicit endpoint getter.
    let (url, key) = store
        .get_provider_endpoint(id1)
        .expect("get endpoint")
        .unwrap();
    assert_eq!(url, "https://api.openai.com");
    assert_eq!(key, "sk-abcd1234");

    // Update.
    store
        .update_provider(id2, "OpenRouter", "https://openrouter.ai", "or-newkey9999", "openai")
        .expect("update");
    let (_, key2) = store.get_provider_endpoint(id2).expect("get").unwrap();
    assert_eq!(key2, "or-newkey9999");

    // Delete.
    store.delete_provider(id1).expect("delete");
    let after = store.list_providers().expect("list");
    assert_eq!(after.len(), 1);
    assert!(store.get_provider(id1).expect("get").is_none());
}

#[test]
fn provider_usage_window() {
    let (_tmp, store) = fresh_store();
    let pid = store
        .create_provider("P", "https://x", "k", "openai")
        .expect("create");

    store.record_provider_usage(pid, 100, 50).expect("record");
    store.record_provider_usage(pid, 200, 80).expect("record");

    // Last hour sums in + out.
    let total = store.provider_usage_last_hour(pid).expect("usage");
    assert_eq!(total, (100 + 50) + (200 + 80));
}

#[tokio::test]
async fn dispatch_route_switches_between_local_and_cloud() {
    // A dispatch provider pointing at a bogus local URL; we never rely on a live
    // server here — only inspect the route + list_models routing behavior. The
    // embedded engine uses a temp models dir (an empty catalog is fine).
    let tmp = tempfile::tempdir().expect("tempdir");
    let embedded = phoenix_agent::model::ambercore_embedded::EmbeddedAmberCore::new(Some(
        tmp.path().to_path_buf(),
    ))
    .expect("embedded engine");
    let provider = DispatchProvider::new(
        "http://localhost:11434",
        ActiveRoute::Local {
            backend: LocalBackend::Ollama,
        },
        embedded,
    );

    // Starts local-ollama.
    match provider.route().await {
        ActiveRoute::Local { backend } => assert_eq!(backend, LocalBackend::Ollama),
        _ => panic!("expected local route"),
    }

    // Switch to AmberCore (local) — over HTTP (remote)…
    provider
        .set_local(LocalBackend::AmberCore, "http://localhost:42069".into())
        .await;
    match provider.route().await {
        ActiveRoute::Local { backend } => assert_eq!(backend, LocalBackend::AmberCore),
        _ => panic!("expected ambercore route"),
    }

    // …then to the embedded engine: same route shape, and list_models answers
    // from the in-process catalog (empty) instead of probing the network.
    provider.set_local_embedded().await;
    match provider.route().await {
        ActiveRoute::Local { backend } => assert_eq!(backend, LocalBackend::AmberCore),
        _ => panic!("expected ambercore route (embedded)"),
    }
    let tags = provider.list_models().await.expect("embedded list_models");
    assert!(tags.is_empty());

    // Switch to a cloud provider.
    provider
        .set_cloud(CloudRoute {
            provider_id: 7,
            base_url: "https://api.openai.com".into(),
            api_key: "sk-test".into(),
        })
        .await;
    match provider.route().await {
        ActiveRoute::Cloud { route } => {
            assert_eq!(route.provider_id, 7);
            assert_eq!(route.base_url, "https://api.openai.com");
        }
        _ => panic!("expected cloud route"),
    }

    // list_models() now routes to cloud — which tries to hit the network and
    // fails (no real server / no network in CI). We assert it targets the cloud
    // URL (not the local one) by inspecting the error.
    let err = provider.list_models().await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("openai.com") || msg.contains("api.openai"),
        "cloud list_models should target the provider URL, got: {msg}"
    );
}

/// URL helpers for the AmberCore pull — filename derivation (query strings
/// must not leak into filenames) and tokenizer-URL derivation.
mod pull_urls {
    use phoenix_agent::web::model_urls::{filename_from_url, hf_file_url, tokenizer_candidates};

    #[test]
    fn filename_strips_query_and_adds_gguf_extension() {
        // HF "copy download link" appends ?download=true — it must not become
        // part of the filename (it used to).
        assert_eq!(
            filename_from_url(
                "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true"
            )
            .as_deref(),
            Some("Qwen3-8B-Q4_K_M.gguf")
        );
        // Missing extension is added; a plain URL is untouched.
        assert_eq!(
            filename_from_url("https://example.com/models/qwen2-0_5b").as_deref(),
            Some("qwen2-0_5b.gguf")
        );
        assert_eq!(
            filename_from_url("https://example.com/m/Model.Q4_K_M.GGUF").as_deref(),
            Some("Model.Q4_K_M.GGUF")
        );
        // No usable last segment.
        assert_eq!(filename_from_url("https://example.com/"), None);
    }

    #[test]
    fn hf_url_normalizes_host_and_blob_links() {
        // /blob/ page links must become /resolve/ download links.
        assert_eq!(
            hf_file_url("https://hf.co/bartowski/Qwen_Qwen2-0.5B-Instruct-GGUF/blob/main/qwen2-0.5b-instruct-q4_k_m.gguf").as_deref(),
            Some("https://huggingface.co/bartowski/Qwen_Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0.5b-instruct-q4_k_m.gguf")
        );
        // Already-direct links pass through, host canonicalized.
        assert_eq!(
            hf_file_url("https://www.huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/x.gguf").as_deref(),
            Some("https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/x.gguf")
        );
        // Non-HF hosts are left to the caller (None).
        assert_eq!(hf_file_url("https://example.com/file.gguf"), None);
    }

    #[test]
    fn tokenizer_candidates_prefer_same_repo_then_base_repo() {
        // Same repo + revision first, then the base model repo (GGUF suffix
        // stripped, case-insensitive).
        assert_eq!(
            tokenizer_candidates("https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"),
            vec![
                "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/tokenizer.json",
                "https://huggingface.co/Qwen/Qwen3-8B/resolve/main/tokenizer.json",
            ]
        );
        // A non-default revision is preserved for the primary candidate.
        let cands = tokenizer_candidates("https://huggingface.co/o/r-gguf/resolve/v1.0/file.gguf");
        assert_eq!(
            cands,
            vec![
                "https://huggingface.co/o/r-gguf/resolve/v1.0/tokenizer.json",
                "https://huggingface.co/o/r/resolve/main/tokenizer.json",
            ]
        );
        // Repo without a GGUF suffix has no base-repo fallback.
        assert_eq!(
            tokenizer_candidates("https://huggingface.co/o/repo/resolve/main/file.gguf"),
            vec!["https://huggingface.co/o/repo/resolve/main/tokenizer.json"]
        );
        // Non-HF URLs can't derive anything.
        assert!(tokenizer_candidates("https://example.com/file.gguf").is_empty());
    }

    #[test]
    fn folder_name_sanitizes_and_uses_the_stem() {
        use phoenix_agent::web::model_urls::model_folder_name;
        // Normal case: the GGUF stem becomes the folder.
        assert_eq!(model_folder_name("gemma3-1b-q4_k_m.gguf"), "gemma3-1b-q4_k_m");
        // Path-hostile characters become dashes; case-insensitive extension.
        assert_eq!(model_folder_name("Model<1?:.GGUF"), "Model-1");
        // Degenerate names fall back instead of empty/dot folders.
        assert_eq!(model_folder_name("???.gguf"), "model");
        assert_eq!(model_folder_name(".gguf"), "model");
    }

    #[test]
    fn split_gguf_shards_are_detected() {
        use phoenix_agent::web::model_urls::is_split_gguf;
        // The standard sharded pattern.
        assert!(is_split_gguf("Mixtral-8x7B-Instruct-v0.1-Q4_K_M-00001-of-00003.gguf"));
        assert!(is_split_gguf("model-00002-of-00009.gguf"));
        // Benign "-of-" in names must NOT trip it.
        assert!(!is_split_gguf("best-of-7b.gguf"));
        assert!(!is_split_gguf("state-of-the-art-q4.gguf"));
        // Plain single files.
        assert!(!is_split_gguf("gemma3-1b-q4.gguf"));
    }
}
