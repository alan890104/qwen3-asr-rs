use anyhow::Context;
use log::info;
use std::path::Path;

pub(crate) fn hf_url(model_id: &str, filename: &str) -> String {
    format!("https://huggingface.co/{}/resolve/main/{}", model_id, filename)
}

/// Make a GET request; returns `None` on 404, error on other failures.
pub(crate) fn hf_try_get(url: &str) -> anyhow::Result<Option<reqwest::blocking::Response>> {
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;
    let mut b = client.get(url);
    if let Ok(tok) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        b = b.header("Authorization", format!("Bearer {}", tok));
    }
    let resp = b.send()?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {} for {}", resp.status(), url);
    }
    Ok(Some(resp))
}

/// GET a URL and return the full body as bytes.
pub(crate) fn hf_get_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
    hf_try_get(url)?
        .ok_or_else(|| anyhow::anyhow!("404: {}", url))
        .and_then(|r| Ok(r.bytes()?.to_vec()))
}

/// Stream a URL to a file, printing progress to stderr.
pub(crate) fn hf_stream_to_file(url: &str, path: &std::path::Path) -> anyhow::Result<()> {
    use std::io::{Read, Write};
    info!("Downloading {}", url);
    let client = reqwest::blocking::Client::builder().timeout(None).build()?;
    let mut b = client.get(url);
    if let Ok(tok) = std::env::var("HUGGING_FACE_HUB_TOKEN") {
        b = b.header("Authorization", format!("Bearer {}", tok));
    }
    let mut resp = b.send()?;
    if !resp.status().is_success() {
        anyhow::bail!("HTTP {} for {}", resp.status(), url);
    }
    let mut file = std::fs::File::create(path)?;
    let mut downloaded = 0u64;
    let mut buf = [0u8; 65536];
    loop {
        let n = resp.read(&mut buf)?;
        if n == 0 { break; }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
    }
    info!("Downloaded {:.1} MB", downloaded as f64 / 1_048_576.0);
    Ok(())
}

/// Ensure model files for `model_id` exist under `cache_dir` and return the
/// model directory path.  Downloads from HuggingFace only when needed.
///
/// Cache layout: `{cache_dir}/{model_id.replace('/', '--')}/`
/// A `.complete` marker file signals that all files are present. If the
/// directory exists but `.complete` is missing (interrupted download), the
/// directory is removed and the download restarts.
pub(crate) fn ensure_model_cached(model_id: &str, cache_dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    let sanitized = model_id.replace('/', "--");
    let model_dir = cache_dir.join(&sanitized);
    let marker = model_dir.join(".complete");

    // Fast path: already downloaded.
    if marker.exists() {
        info!("Using cached model at {}", model_dir.display());
        return Ok(model_dir);
    }

    // Partial / interrupted download — remove and restart.
    if model_dir.exists() {
        info!("Removing incomplete download at {}", model_dir.display());
        std::fs::remove_dir_all(&model_dir)?;
    }

    info!("Downloading '{}' from HuggingFace to {}…", model_id, model_dir.display());
    std::fs::create_dir_all(&model_dir)?;

    // config.json
    let config_bytes = hf_get_bytes(&hf_url(model_id, "config.json"))
        .context("download config.json")?;
    std::fs::write(model_dir.join("config.json"), &config_bytes)?;

    // Weights: check for sharded index first.
    if let Some(resp) = hf_try_get(&hf_url(model_id, "model.safetensors.index.json"))? {
        let index_text = resp.text()?;
        std::fs::write(model_dir.join("model.safetensors.index.json"), &index_text)?;

        let index: serde_json::Value = serde_json::from_str(&index_text)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("invalid model.safetensors.index.json"))?;
        let shards: std::collections::HashSet<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        for shard in &shards {
            hf_stream_to_file(&hf_url(model_id, shard), &model_dir.join(shard))
                .with_context(|| format!("download shard {}", shard))?;
        }
    } else {
        hf_stream_to_file(
            &hf_url(model_id, "model.safetensors"),
            &model_dir.join("model.safetensors"),
        )
        .context("download model.safetensors")?;
    }

    // Tokenizer: Qwen3-ASR ships tokenizer_config.json (with added_tokens_decoder)
    // but not tokenizer.json. Reconstruct from vocab.json + merges.txt + config.
    let tok_config = String::from_utf8(
        hf_get_bytes(&hf_url(model_id, "tokenizer_config.json"))
            .context("download tokenizer_config.json")?,
    )?;
    let vocab = String::from_utf8(
        hf_get_bytes(&hf_url(model_id, "vocab.json")).context("download vocab.json")?,
    )?;
    let merges = String::from_utf8(
        hf_get_bytes(&hf_url(model_id, "merges.txt")).context("download merges.txt")?,
    )?;
    let tok_json = build_qwen3_tokenizer_json(&vocab, &merges, &tok_config)?;
    std::fs::write(model_dir.join("tokenizer.json"), tok_json)?;

    // Mark download as complete.
    std::fs::write(&marker, b"")?;
    info!("Download complete, cached at {}", model_dir.display());

    Ok(model_dir)
}

/// Build the Qwen3 tokenizer JSON from vocab.json, merges.txt, and tokenizer_config.json.
/// The added_tokens list is derived from tokenizer_config.json's added_tokens_decoder field,
/// so no special tokens need to be hardcoded here.
fn build_qwen3_tokenizer_json(vocab: &str, merges: &str, tok_config: &str) -> anyhow::Result<Vec<u8>> {
    let vocab_val: serde_json::Value = serde_json::from_str(vocab)?;
    let merges_vec: Vec<&str> = merges
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .collect();

    // Build added_tokens from tokenizer_config.json's added_tokens_decoder.
    let tok_cfg: serde_json::Value = serde_json::from_str(tok_config)?;
    let mut added_tokens: Vec<serde_json::Value> = Vec::new();
    if let Some(decoder_map) = tok_cfg["added_tokens_decoder"].as_object() {
        let mut entries: Vec<(u64, &serde_json::Value)> = decoder_map
            .iter()
            .filter_map(|(k, v)| k.parse::<u64>().ok().map(|id| (id, v)))
            .collect();
        entries.sort_by_key(|(id, _)| *id);
        for (id, v) in &entries {
            added_tokens.push(serde_json::json!({
                "id": id,
                "content": v["content"],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": v["special"]
            }));
        }
    }
    let added_tokens = serde_json::Value::Array(added_tokens);

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
                    "behavior": "Isolated",
                    "invert": false
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": false,
                    "trim_offsets": false,
                    "use_regex": false
                }
            ]
        },
        "post_processor": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": false,
            "use_regex": false
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": vocab_val,
            "merges": merges_vec
        }
    });

    serde_json::to_vec(&tokenizer_json).map_err(Into::into)
}
