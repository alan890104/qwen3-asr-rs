//! Streaming verification: compare batch vs streaming transcription on real audio.
//!
//! Usage:
//!   cargo run --release --example streaming_verify -- <wav_file> [chunk_size_sec]
//!
//! Example:
//!   cargo run --release --example streaming_verify -- tests/fixtures/audio/sample4.wav 2.0

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <wav_file> [chunk_size_sec]", args[0]);
        std::process::exit(1);
    }

    let wav_path = &args[1];
    let chunk_size_sec: f32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2.0);

    // ── Load model ───────────────────────────────────────────────────────
    let model_dir = std::env::var("QWEN3_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .expect("Set QWEN3_ASR_MODEL_DIR to the model directory");

    println!("Loading model from: {}", model_dir.display());
    let t0 = Instant::now();
    let device = qwen3_asr::best_device();
    let engine = qwen3_asr::AsrInference::load(&model_dir, device)
        .expect("failed to load model");
    println!("Model loaded in {:.2}s\n", t0.elapsed().as_secs_f64());

    // ── Load audio ───────────────────────────────────────────────────────
    let samples = qwen3_asr::load_audio_wav(wav_path, 16000).expect("failed to load wav");
    let duration_sec = samples.len() as f32 / 16000.0;
    println!("Audio: {} ({:.1}s, {} samples)", wav_path, duration_sec, samples.len());
    println!("Chunk size: {:.1}s ({} samples)\n", chunk_size_sec, (chunk_size_sec * 16000.0) as usize);

    // ── Batch transcription ──────────────────────────────────────────────
    println!("═══ BATCH TRANSCRIPTION ═══");
    let t0 = Instant::now();
    let batch_result = engine
        .transcribe_samples(&samples, qwen3_asr::TranscribeOptions::default())
        .expect("batch transcription failed");
    let batch_elapsed = t0.elapsed();
    println!("Text: {}", batch_result.text);
    println!("Language: {}", batch_result.language);
    println!("Time: {:.2}s (RTF: {:.3}x)\n", batch_elapsed.as_secs_f64(),
             batch_elapsed.as_secs_f64() / duration_sec as f64);

    // ── Streaming transcription ──────────────────────────────────────────
    println!("═══ STREAMING TRANSCRIPTION ═══");
    let stream_opts = qwen3_asr::StreamingOptions::default()
        .with_chunk_size_sec(chunk_size_sec);
    let mut state = engine.init_streaming(stream_opts);

    let chunk_samples = (chunk_size_sec * 16000.0) as usize;
    let mut step = 0;
    let t0 = Instant::now();
    let mut step_times: Vec<f64> = Vec::new();

    for chunk_start in (0..samples.len()).step_by(chunk_samples) {
        let chunk_end = (chunk_start + chunk_samples).min(samples.len());
        let chunk = &samples[chunk_start..chunk_end];
        let step_t0 = Instant::now();

        match engine.feed_audio(&mut state, chunk) {
            Ok(Some(result)) => {
                step += 1;
                let step_elapsed = step_t0.elapsed().as_secs_f64();
                step_times.push(step_elapsed);
                let audio_time = chunk_end as f64 / 16000.0;
                println!(
                    "  Step {:2} | audio={:.1}s | time={:.3}s | text: {}",
                    step, audio_time, step_elapsed, result.text
                );
            }
            Ok(None) => {
                let chunk_dur = chunk.len() as f64 / 16000.0;
                println!(
                    "  (buffered {:.2}s, not enough for a chunk)",
                    chunk_dur
                );
            }
            Err(e) => {
                eprintln!("  ERROR at step {}: {}", step + 1, e);
                break;
            }
        }
    }

    // Final flush
    println!("\n  --- finish_streaming ---");
    let final_t0 = Instant::now();
    let final_result = engine.finish_streaming(&mut state).expect("finish_streaming failed");
    let final_elapsed = final_t0.elapsed().as_secs_f64();
    let total_elapsed = t0.elapsed();

    println!("  Final text: {}", final_result.text);
    println!("  Final lang: {}", final_result.language);
    println!("  Final time: {:.3}s", final_elapsed);

    // ── Summary ──────────────────────────────────────────────────────────
    println!("\n═══ SUMMARY ═══");
    println!("Audio duration: {:.1}s", duration_sec);
    println!("Batch:   \"{}\"\n  time={:.2}s  RTF={:.3}x",
             batch_result.text, batch_elapsed.as_secs_f64(),
             batch_elapsed.as_secs_f64() / duration_sec as f64);
    println!("Stream:  \"{}\"\n  time={:.2}s  RTF={:.3}x  steps={}",
             final_result.text, total_elapsed.as_secs_f64(),
             total_elapsed.as_secs_f64() / duration_sec as f64, step);

    if !step_times.is_empty() {
        let avg = step_times.iter().sum::<f64>() / step_times.len() as f64;
        let max = step_times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("  Avg step time: {:.3}s  Max: {:.3}s", avg, max);
    }

    // Compare
    let match_status = if batch_result.text == final_result.text {
        "EXACT MATCH"
    } else {
        "DIFFER"
    };
    println!("\nBatch vs Stream: {}", match_status);
}
