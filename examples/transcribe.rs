//! Transcribe a WAV file using a model from HuggingFace Hub.
//!
//! Run:
//!   cargo run --example transcribe --features hub --release -- tests/fixtures/audio/sample1.wav
//!
//! Set MODEL_ID env var to switch model (default: Qwen/Qwen3-ASR-0.6B).
//! Set CACHE_DIR env var to change cache location (default: models/).

use anyhow::Result;
use qwen3_asr::{AsrInference, TranscribeOptions};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let wav_path = args.get(1).map(String::as_str).unwrap_or("tests/fixtures/audio/sample1.wav");
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "Qwen/Qwen3-ASR-0.6B".to_string());
    let cache_dir = std::env::var("CACHE_DIR")
        .unwrap_or_else(|_| "models".to_string());

    let device = qwen3_asr::best_device();
    eprintln!("Device    : {device:?}");
    eprintln!("Model     : {model_id}");
    eprintln!("Cache dir : {cache_dir}");

    let engine = AsrInference::from_pretrained(&model_id, Path::new(&cache_dir), device)?;
    let result = engine.transcribe(wav_path, TranscribeOptions::default())?;

    println!("Language : {}", result.language);
    println!("Text     : {}", result.text);
    Ok(())
}
