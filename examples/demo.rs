use qwen3_asr::{AsrInference, TranscribeOptions};

use anyhow::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    // CLI argument parsing: optional positional argument for model directory.
    // Usage: cargo run --example demo --release -- [model_dir]
    let args: Vec<String> = std::env::args().collect();
    let model_dir_arg: Option<String> = args.get(1).filter(|s| !s.starts_with('-')).cloned();

    let audio_dir = PathBuf::from("tests/fixtures/audio");

    let device = qwen3_asr::best_device();
    eprintln!("Device: {:?}", device);

    let model_dir = model_dir_arg
        .map(PathBuf::from)
        .or_else(|| std::env::var("MODEL_DIR").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("models"));
    eprintln!("Model dir: {}", model_dir.display());
    let engine = AsrInference::load(&model_dir, device)?;

    // Short samples: expected text is loaded from the paired .txt file.
    // These are used for exact-match regression testing.
    let short_samples: &[&str] = &["sample1.wav", "sample2.wav"];

    // Long samples: transcribed and shown alongside the reference .txt,
    // but not exact-matched (different model sizes may differ slightly).
    let long_samples: &[&str] = &["sample4.wav", "sample5.wav", "sample6.wav"];

    let mut pass = 0;
    let mut total = 0;

    eprintln!("\n{}", "─".repeat(60));
    eprintln!("SHORT SAMPLES (exact match)");
    for wav_file in short_samples {
        total += 1;
        let audio_path = audio_dir.join(wav_file);
        let txt_path = audio_path.with_extension("txt");

        let expected = std::fs::read_to_string(&txt_path)
            .unwrap_or_else(|_| panic!("Missing groundtruth: {}", txt_path.display()));
        let expected = expected.trim();

        eprintln!("\n{}", "=".repeat(60));
        eprintln!("Testing: {}", wav_file);

        let path_str = audio_path.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {}", audio_path.display()))?;
        match engine.transcribe(path_str, TranscribeOptions::default()) {
            Ok(result) => {
                let got = result.text.trim();
                let matched = got == expected;

                println!("File    : {}", wav_file);
                println!("Expected: {}", expected);
                println!("Got     : {}", got);
                println!("Language: {}", result.language);
                println!("Status  : {}", if matched { "PASS ✓" } else { "FAIL ✗" });

                if matched {
                    pass += 1;
                } else {
                    println!("(raw output: {:?})", result.raw_output);
                }
            }
            Err(e) => {
                println!("File    : {}", wav_file);
                println!("ERROR   : {}", e);
                println!("Status  : FAIL ✗");
            }
        }
        println!();
    }

    eprintln!("\n{}", "─".repeat(60));
    eprintln!("LONG SAMPLES (transcription + reference)");
    for wav_file in long_samples {
        let audio_path = audio_dir.join(wav_file);
        if !audio_path.exists() {
            eprintln!("Skipping {} (not found)", wav_file);
            continue;
        }
        let txt_path = audio_path.with_extension("txt");
        let reference = std::fs::read_to_string(&txt_path).ok();

        eprintln!("\n{}", "=".repeat(60));
        eprintln!("Transcribing: {}", wav_file);

        let path_str = audio_path.to_str()
            .ok_or_else(|| anyhow::anyhow!("non-UTF8 path: {}", audio_path.display()))?;
        match engine.transcribe(path_str, TranscribeOptions::default()) {
            Ok(result) => {
                println!("File     : {}", wav_file);
                if let Some(ref r) = reference {
                    println!("Reference: {}", r.trim());
                }
                println!("Got      : {}", result.text.trim());
                println!("Language : {}", result.language);
            }
            Err(e) => {
                println!("File    : {}", wav_file);
                println!("ERROR   : {}", e);
            }
        }
        println!();
    }

    println!("Results: {}/{} passed", pass, total);
    if pass == total {
        println!("All tests passed!");
    }

    Ok(())
}
