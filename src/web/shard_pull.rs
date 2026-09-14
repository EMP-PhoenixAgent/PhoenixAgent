//! Sharded-safetensors pull flow — included by `commands.rs`.
//!
//! A sharded HF repo ships `model-00001-of-0000N.safetensors` shards plus an
//! `index.json` (tensor → shard map). This flow downloads index.json + every
//! shard (one aggregated progress bar over summed byte counts) + config.json
//! + tokenizer, then registers the FIRST shard as the catalog entry — the
//! engine's multi-region reader stitches the rest via the sibling index.json.

use super::commands::download_to_file;

#[allow(clippy::too_many_arguments)]
pub(super) async fn pull_sharded_safetensors(
    state: &tauri::State<'_, crate::web::state::WebState>,
    models_dir: &std::path::Path,
    model_url: &str,
    first_shard: &str,
    base_stem: &str,
    repo_name: Option<String>,
    repo_base: &str,
    pull_id: &str,
    app: &tauri::AppHandle,
) -> Result<String, String> {
    use tauri::Emitter;
    let emit = |completed: u64, total: Option<u64>| {
        let payload = serde_json::json!({
            "id": pull_id, "phase": "model", "completed": completed, "total": total,
        });
        let _ = app.emit("ambercore-pull-progress", payload);
    };

    let folder = super::model_urls::model_folder_name(base_stem);
    let model_dir = models_dir.join(&folder);
    std::fs::create_dir_all(&model_dir)
        .map_err(|e| format!("create model folder {}: {e}", model_dir.display()))?;

    // 1. index.json — the tensor → shard map (also stored beside the weights;
    //    the engine reads it at load time).
    let client = reqwest::Client::builder()
        .build()
        .map_err(|e| format!("build client: {e}"))?;
    let index_url = format!("{}/resolve/main/index.json", repo_base.trim_end_matches('/'));
    let index_raw = client
        .get(&index_url)
        .send()
        .await
        .map_err(|e| format!("fetch index.json: {e}"))?
        .error_for_status()
        .map_err(|e| format!("index.json: {e}"))?
        .text()
        .await
        .map_err(|e| format!("read index.json: {e}"))?;
    let index: serde_json::Value = serde_json::from_str(&index_raw)
        .map_err(|e| format!("parse index.json: {e}"))?;
    let mut shards: Vec<String> = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .map(|wm| {
            wm.values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    shards.sort();
    shards.dedup();
    if shards.is_empty() {
        return Err("index.json has no weight_map entries".into());
    }
    std::fs::write(model_dir.join("index.json"), &index_raw)
        .map_err(|e| format!("write index.json: {e}"))?;

    // 2. Shards — skip ones already on disk; one bar over the summed bytes.
    let mut sizes: Vec<Option<u64>> = Vec::with_capacity(shards.len());
    for shard in &shards {
        let dest = model_dir.join(shard);
        if dest.is_file() && std::fs::metadata(&dest).map(|m| m.len() > 0).unwrap_or(false) {
            sizes.push(std::fs::metadata(&dest).ok().map(|m| m.len()));
            continue;
        }
        // Content-Length via HEAD (redirects resolved by the client).
        let url = format!("{}/resolve/main/{}", repo_base.trim_end_matches('/'), shard);
        let head = client.head(&url).send().await.ok();
        sizes.push(head.and_then(|r| r.content_length()));
    }
    let known_total: Option<u64> = {
        let any_missing = sizes.iter().any(|s| s.is_none());
        if any_missing {
            None
        } else {
            sizes.iter().try_fold(0u64, |acc, s| s.map(|v| acc + v))
        }
    };
    let mut done_before: u64 = 0;
    for shard in &shards {
        let dest = model_dir.join(shard);
        let already = dest.is_file()
            && std::fs::metadata(&dest).map(|m| m.len() > 0).unwrap_or(false);
        if already {
            done_before += std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            emit(done_before, known_total);
            tracing::info!(shard, "shard already on disk, skipping");
            continue;
        }
        let url = format!("{}/resolve/main/{}", repo_base.trim_end_matches('/'), shard);
        let base = done_before;
        let total_all = known_total;
        let emit_shard = move |_phase: &'static str, completed: u64, _t: Option<u64>| {
            emit(base + completed, total_all);
        };
        download_to_file(&client, &url, &dest, "model", &emit_shard).await?;
        done_before += std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    }

    // 3. config.json — required by the F16 loader.
    let cfg_dest = model_dir.join("config.json");
    if !cfg_dest.is_file() {
        let cfg_url = format!("{}/resolve/main/config.json", repo_base.trim_end_matches('/'));
        let bytes = client
            .get(&cfg_url)
            .send()
            .await
            .map_err(|e| format!("fetch config.json: {e}"))?
            .error_for_status()
            .map_err(|e| format!("config.json: {e}"))?
            .bytes()
            .await
            .map_err(|e| format!("read config.json: {e}"))?;
        std::fs::write(&cfg_dest, &bytes).map_err(|e| format!("write config.json: {e}"))?;
    }

    // 4. Architecture validation — same contract as the single-file path.
    let entry = model_dir.join(first_shard);
    let arch = ambercore::model::gguf::probe_arch(&entry)
        .map_err(|e| format!("downloaded model is not loadable: {e}"))?;
    if !ambercore::model::registry::is_supported(&arch) {
        return Err(format!(
            "AmberCore can't run the `{arch}` architecture yet (supported: {}). \
             The model was kept at {} — it will work here if support is added later.",
            ambercore::model::registry::SUPPORTED_ARCHS.join(", "),
            model_dir.display(),
        ));
    }

    // 5. Tokenizer — same candidates as the single-file flow.
    let stem = entry
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_string();
    let tok_dest = model_dir.join(format!("{stem}.tokenizer.json"));
    if !(tok_dest.is_file() || model_dir.join("tokenizer.json").is_file()) {
        let candidates = super::model_urls::tokenizer_candidates(model_url);
        let mut failures: Vec<String> = Vec::new();
        for cand in &candidates {
            let ok = download_to_file(&client, cand, &tok_dest, "tokenizer", &|_p, _c, _t| {})
                .await
                .is_ok();
            if ok {
                break;
            }
            let _ = tokio::fs::remove_file(&tok_dest).await;
            failures.push(cand.clone());
        }
        if !tok_dest.is_file() {
            return Err(format!(
                "shards downloaded to {}, but no tokenizer was found (tried: {})",
                model_dir.display(),
                failures.join(", ")
            ));
        }
    }

    // 6. Register the FIRST shard; the engine stitches the rest at load time.
    let tag = repo_name.clone().unwrap_or_else(|| base_stem.to_string());
    let tag = if tag.contains('-') { tag } else { format!("{tag}:latest") };
    let rel_path = format!("{folder}/{first_shard}");
    state
        .provider
        .embedded()
        .register_model(&tag, &rel_path)
        .await
        .map_err(|e| format!("register model: {e}"))?;
    tracing::info!(%tag, path = %rel_path, "registered sharded F16 model");
    Ok(tag)
}
