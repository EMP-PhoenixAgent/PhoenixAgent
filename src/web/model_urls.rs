//! URL helpers for the AmberCore model pull.
//!
//! AmberCore loads a model's tokenizer from a sibling `tokenizer.json`
//! (HF format), so a pull is only complete when **both** the GGUF and a
//! tokenizer land in the models directory. These pure functions derive the
//! local filename and candidate tokenizer URLs from the pasted model URL.
//! They are unit-tested in `tests/models_panel.rs`.

/// Derive the local GGUF filename from a model URL: the last path segment with
/// any query string / fragment stripped (Hugging Face's "copy download link"
/// appends `?download=true`, which must not become part of the filename), and
/// a `.gguf` extension added if missing. Returns `None` when the URL has no
/// usable last segment (caller supplies a timestamped fallback name).
pub fn filename_from_url(url: &str) -> Option<String> {
    let clean = strip_query_fragment(url.trim());
    let last = clean.rsplit('/').next().unwrap_or("");
    if last.is_empty() {
        return None;
    }
    if last.to_ascii_lowercase().ends_with(".gguf") {
        Some(last.to_string())
    } else {
        Some(format!("{last}.gguf"))
    }
}

/// Normalize a Hugging Face URL into a direct file-download URL:
/// - accepts `huggingface.co` / `www.huggingface.co` / `hf.co` hosts,
/// - canonicalizes the host to `https://huggingface.co`,
/// - converts `/blob/` page URLs to `/resolve/` download URLs (downloading a
///   blob link yields HTML, not the file).
/// Query strings are preserved (they may matter for the request). Returns
/// `None` for non-HF URLs — the caller uses those as-is.
pub fn hf_file_url(url: &str) -> Option<String> {
    let url = url.trim();
    let after_scheme = url.split_once("://")?.1;
    // The authority runs to the first '/'; HF links carry no userinfo/port.
    let (authority, path) = after_scheme
        .split_once('/')
        .map(|(a, p)| (a, p.to_string()))
        .unwrap_or((after_scheme, String::new()));
    let host = authority.to_ascii_lowercase();
    if !matches!(host.as_str(), "huggingface.co" | "www.huggingface.co" | "hf.co") {
        return None;
    }
    let path = path.replace("/blob/", "/resolve/");
    Some(format!("https://huggingface.co/{path}"))
}

/// Derive candidate `tokenizer.json` URLs for a Hugging Face model URL, best
/// first:
/// 1. Same repo, same revision — most GGUF repos ship a `tokenizer.json`.
/// 2. The base model repo (a trailing `-GGUF`/`_GGUF` suffix stripped from the
///    repo name) at `main` — for GGUF-only repos whose tokenizer lives with
///    the original model.
/// Returns an empty vec for non-HF URLs (no derivation is possible; the user
/// must supply a tokenizer URL explicitly).
pub fn tokenizer_candidates(model_url: &str) -> Vec<String> {
    let Some(hf) = hf_file_url(model_url) else {
        return Vec::new();
    };
    let path = hf
        .strip_prefix("https://huggingface.co/")
        .unwrap_or_default();
    let path = strip_query_fragment(path);
    let segs: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    // Expected shape: {owner}/{repo}/resolve/{rev}/{file…}
    if segs.len() < 4 || (segs[2] != "resolve" && segs[2] != "blob") {
        return Vec::new();
    }
    let (owner, repo, rev) = (segs[0], segs[1], segs[3]);
    let mut out = vec![format!(
        "https://huggingface.co/{owner}/{repo}/resolve/{rev}/tokenizer.json"
    )];
    if let Some(base) = strip_gguf_suffix(repo) {
        out.push(format!(
            "https://huggingface.co/{owner}/{base}/resolve/main/tokenizer.json"
        ));
    }
    out
}

/// Strip a trailing `-GGUF` / `_GGUF` marker (case-insensitive) from a repo
/// name: `Qwen3-8B-GGUF` → `Qwen3-8B`, `Qwen_Qwen2-0.5B-GGUF` → `Qwen_Qwen2-0.5B`.
fn strip_gguf_suffix(repo: &str) -> Option<&str> {
    for sep in ['-', '_'] {
        if let Some(idx) = repo.rfind(sep) {
            if repo[idx + 1..].eq_ignore_ascii_case("gguf") {
                return Some(&repo[..idx]);
            }
        }
    }
    None
}

/// Derive the **per-model folder name** for a pull: the GGUF filename's stem,
/// sanitized into a safe single path segment. Pulls land in
/// `<models_dir>/<folder>/<gguf>` with the tokenizer beside it, so every model
/// owns an isolated folder and models with different vocabularies can never
/// pick up each other's tokenizer.
///
/// Sanitization replaces characters that are illegal (or hostile) in Windows
/// path segments with `-`, trims leading/trailing dots+spaces, and caps the
/// length. `"Model?.gguf"` → `"Model-"`.
pub fn model_folder_name(filename: &str) -> String {
    let stem = filename
        .strip_suffix(".gguf")
        .or_else(|| filename.strip_suffix(".GGUF"))
        .unwrap_or(filename);
    let sanitized: String = stem
        .chars()
        .map(|c| match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '-',
            c if c.is_control() => '-',
            c => c,
        })
        .collect();
    let trimmed = sanitized.trim_matches(['.', ' ', '-']).to_string();
    if trimmed.is_empty() {
        return "model".to_string();
    }
    // Cap the length on a char boundary (Windows MAX_PATH headroom).
    if trimmed.chars().count() > 80 {
        trimmed.chars().take(80).collect()
    } else {
        trimmed
    }
}

/// Detect a **split/sharded GGUF** filename (`…-00001-of-00003.gguf`). AmberCore
/// loads single-file GGUFs only — shards of big MoE models (mixtral, qwen3moe)
/// are published this way — so pulls must fail fast with guidance instead of
/// downloading a partial model that can never load.
pub fn is_split_gguf(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    let base = lower
        .strip_suffix(".gguf")
        .unwrap_or(&lower);
    let Some(idx) = base.find("-of-") else {
        return false;
    };
    let left = &base[..idx];
    let right = &base[idx + "-of-".len()..];
    let left_num = left.rsplit('-').next().unwrap_or("");
    // Both sides must be the shard counters (≥4 digits, all numeric) — e.g.
    // `model-00001-of-00002`. Guards against innocuous "-of-" in names
    // ("best-of-7b" → left_num "best" isn't numeric).
    left_num.len() >= 4
        && left_num.bytes().all(|b| b.is_ascii_digit())
        && right.len() >= 4
        && right.bytes().all(|b| b.is_ascii_digit())
}

/// Strip everything from the first `?` or `#` onwards.
fn strip_query_fragment(s: &str) -> &str {
    let cut = s.find(['?', '#']).unwrap_or(s.len());
    &s[..cut]
}
