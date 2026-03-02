use anyhow::{Context, Result};
use candle_core::quantized::{gguf_file, GgmlDType};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

use crate::config::{AsrConfig, AudioEncoderConfig, RopeScaling, TextDecoderConfig, ThinkerConfig};

// ─── Public API ──────────────────────────────────────────────────────────────

/// Load all tensors from a GGUF file, dequantizing each to a candle Tensor.
///
/// Dtype policy (mirrors the original safetensors BF16 model):
///   - Q8_0 and other quantized types  → dequantize → **BF16**
///     This matches the original BF16 weight dtype so all matmuls in the
///     encoder/decoder run in BF16, enabling Metal's native BF16 acceleration.
///   - F16 (conv weights, embeddings)  → dequantize → **F16** (unchanged)
///   - F32 (norms, biases)             → dequantize → **F32** (unchanged)
///
/// Returns the same `HashMap<String, Tensor>` shape as the safetensors loader,
/// preserving all `thinker.` prefixes so encoder/decoder loaders work unchanged.
pub fn load_gguf_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open GGUF: {}", path.display()))?;

    let content = gguf_file::Content::read(&mut file).context("read GGUF header")?;

    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    let mut weights = HashMap::with_capacity(names.len());

    for name in names {
        let qtensor = content
            .tensor(&mut file, &name, device)
            .with_context(|| format!("read tensor: {}", name))?;

        // Quantized weights are dequantized to BF16 to match the original model's
        // native dtype and leverage Metal BF16 GEMM kernels.
        // F16 and F32 tensors are kept as-is (dequantize() is a no-op for them).
        let tensor = match qtensor.dtype() {
            GgmlDType::Q4_0 | GgmlDType::Q4_1
            | GgmlDType::Q5_0 | GgmlDType::Q5_1
            | GgmlDType::Q8_0 | GgmlDType::Q8_1
            | GgmlDType::Q2K | GgmlDType::Q3K
            | GgmlDType::Q4K | GgmlDType::Q5K
            | GgmlDType::Q6K | GgmlDType::Q8K => {
                qtensor
                    .dequantize(device)
                    .with_context(|| format!("dequantize tensor: {}", name))?
                    .to_dtype(DType::BF16)?
            }
            _ => qtensor
                .dequantize(device)
                .with_context(|| format!("dequantize tensor: {}", name))?,
        };

        weights.insert(name, tensor);
    }

    Ok(weights)
}

/// Read the `AsrConfig` and embedded tokenizer JSON string from GGUF metadata.
pub fn load_gguf_config(path: &Path) -> Result<(AsrConfig, String)> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("open GGUF: {}", path.display()))?;

    let content = gguf_file::Content::read(&mut file).context("read GGUF header")?;
    let meta = &content.metadata;

    // ── helpers ───────────────────────────────────────────────────────────────

    let get_u32 = |key: &str, default: u32| -> usize {
        meta.get(key)
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(default) as usize
    };

    let get_f64 = |key: &str, default: f64| -> f64 {
        meta.get(key)
            .and_then(|v| v.to_f64().ok())
            .unwrap_or(default)
    };

    let get_bool = |key: &str, default: bool| -> bool {
        meta.get(key)
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(default)
    };

    // ── audio encoder ─────────────────────────────────────────────────────────
    let audio_config = AudioEncoderConfig {
        d_model:                  get_u32("qwen3_asr.audio.d_model",                 896),
        encoder_layers:           get_u32("qwen3_asr.audio.encoder_layers",          18),
        encoder_attention_heads:  get_u32("qwen3_asr.audio.encoder_attention_heads", 14),
        encoder_ffn_dim:          get_u32("qwen3_asr.audio.encoder_ffn_dim",         3584),
        num_mel_bins:             get_u32("qwen3_asr.audio.num_mel_bins",            128),
        max_source_positions:     get_u32("qwen3_asr.audio.max_source_positions",    1500),
        n_window:                 get_u32("qwen3_asr.audio.n_window",                50),
        n_window_infer:           get_u32("qwen3_asr.audio.n_window_infer",          800),
        conv_chunksize:           get_u32("qwen3_asr.audio.conv_chunksize",          500),
        output_dim:               get_u32("qwen3_asr.audio.output_dim",              1024),
    };

    // ── mrope_section ─────────────────────────────────────────────────────────
    // The Python gguf writer stores Python ints as I32 (not U32), so we try
    // all integer variants and fall back to the default [24, 20, 20].
    let mrope_section: Vec<usize> = meta
        .get("qwen3_asr.text.mrope_section")
        .and_then(|v| v.to_vec().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    v.to_u32()
                        .ok()
                        .or_else(|| v.to_i32().ok().map(|x| x as u32))
                        .or_else(|| v.to_u64().ok().map(|x| x as u32))
                        .or_else(|| v.to_i64().ok().map(|x| x as u32))
                })
                .map(|x| x as usize)
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![24, 20, 20]);

    // ── text decoder ──────────────────────────────────────────────────────────
    let text_config = TextDecoderConfig {
        vocab_size:          get_u32("qwen3_asr.text.vocab_size",          151936),
        hidden_size:         get_u32("qwen3_asr.text.hidden_size",         1024),
        intermediate_size:   get_u32("qwen3_asr.text.intermediate_size",   3072),
        num_hidden_layers:   get_u32("qwen3_asr.text.num_hidden_layers",   28),
        num_attention_heads: get_u32("qwen3_asr.text.num_attention_heads", 16),
        num_key_value_heads: get_u32("qwen3_asr.text.num_key_value_heads", 8),
        head_dim:            get_u32("qwen3_asr.text.head_dim",            128),
        rms_norm_eps:        get_f64("qwen3_asr.text.rms_norm_eps",        1e-6),
        rope_theta:          get_f64("qwen3_asr.text.rope_theta",          1_000_000.0),
        tie_word_embeddings: get_bool("qwen3_asr.text.tie_word_embeddings", true),
        rope_scaling: Some(RopeScaling {
            rope_type: "mrope".to_string(),
            mrope_section,
            interleaved: false,
            mrope_interleaved: true,
        }),
    };

    // ── special token IDs ─────────────────────────────────────────────────────
    let audio_start_token_id =
        meta.get("qwen3_asr.audio_start_token_id")
            .and_then(|v| v.to_u32().ok())
            .map(|x| x as i64)
            .unwrap_or(151669);

    let audio_end_token_id =
        meta.get("qwen3_asr.audio_end_token_id")
            .and_then(|v| v.to_u32().ok())
            .map(|x| x as i64)
            .unwrap_or(151670);

    let audio_token_id =
        meta.get("qwen3_asr.audio_token_id")
            .and_then(|v| v.to_u32().ok())
            .map(|x| x as i64)
            .unwrap_or(151676);

    let thinker_config = ThinkerConfig {
        audio_config,
        text_config,
        audio_start_token_id,
        audio_end_token_id,
        audio_token_id,
    };

    let config = AsrConfig { thinker_config };

    // ── tokenizer JSON ────────────────────────────────────────────────────────
    let tokenizer_json = meta
        .get("tokenizer.huggingface.json")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.clone())
        .context("tokenizer.huggingface.json not found in GGUF metadata")?;

    Ok((config, tokenizer_json))
}
