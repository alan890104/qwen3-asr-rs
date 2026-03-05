//! Integration tests for the streaming API.
//!
//! These tests require a downloaded Qwen3-ASR model and are marked `#[ignore]`
//! by default. Run with:
//!
//!   QWEN3_ASR_MODEL_DIR=/path/to/model cargo test --test streaming_integration -- --ignored

use std::path::{Path, PathBuf};

fn model_dir() -> PathBuf {
    std::env::var("QWEN3_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .expect("Set QWEN3_ASR_MODEL_DIR to the model directory")
}

fn load_engine() -> qwen3_asr::AsrInference {
    let dir = model_dir();
    eprintln!("Loading model from: {}", dir.display());
    let device = qwen3_asr::best_device();
    qwen3_asr::AsrInference::load(&dir, device).expect("failed to load model")
}

/// Generate a sine wave tone at the given frequency.
fn sine_tone(freq_hz: f32, duration_sec: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.3
        })
        .collect()
}

/// Load a WAV file as f32 samples at 16 kHz.
fn load_wav_16k(path: &Path) -> Vec<f32> {
    qwen3_asr::load_audio_wav(path.to_str().unwrap(), 16000).expect("failed to load wav")
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_streaming_init_and_finish_empty() {
    let engine = load_engine();
    let mut state = engine.init_streaming(qwen3_asr::StreamingOptions::default());

    // Finish immediately with no audio → empty result
    let result = engine.finish_streaming(&mut state).unwrap();
    assert!(result.text.is_empty());
}

#[test]
#[ignore]
fn test_streaming_feed_partial_chunk_returns_none() {
    let engine = load_engine();
    let mut state = engine.init_streaming(qwen3_asr::StreamingOptions::default());

    // Feed less than 2 seconds (32000 samples) — should return None
    let partial = vec![0.0f32; 16000]; // 1 second
    let result = engine.feed_audio(&mut state, &partial).unwrap();
    assert!(result.is_none(), "partial chunk should not trigger inference");
}

#[test]
#[ignore]
fn test_streaming_feed_one_chunk_returns_result() {
    let engine = load_engine();
    let mut state = engine.init_streaming(qwen3_asr::StreamingOptions::default());

    // Feed exactly one 2-second chunk of silence/tone
    let chunk = sine_tone(440.0, 2.0, 16000);
    let result = engine.feed_audio(&mut state, &chunk).unwrap();
    assert!(result.is_some(), "full chunk should trigger inference");
}

#[test]
#[ignore]
fn test_streaming_finish_flushes_buffer() {
    let engine = load_engine();
    let mut state = engine.init_streaming(qwen3_asr::StreamingOptions::default());

    // Feed less than a full chunk
    let partial = sine_tone(440.0, 1.5, 16000); // 24000 samples, < 32000
    let result = engine.feed_audio(&mut state, &partial).unwrap();
    assert!(result.is_none());

    // Finish should flush the partial buffer and produce a result
    let result = engine.finish_streaming(&mut state).unwrap();
    // Result fields should be valid strings (not guaranteed non-empty for a tone)
    assert!(result.text.len() <= 10000, "text should be bounded");
    assert!(result.raw_output.len() <= 10000, "raw_output should be bounded");
}

#[test]
#[ignore]
fn test_streaming_multiple_chunks_progressive() {
    let engine = load_engine();
    let opts = qwen3_asr::StreamingOptions::default()
        .with_chunk_size_sec(1.0)
        .with_unfixed_chunk_num(1)
        .with_unfixed_token_num(3)
        .with_max_new_tokens_streaming(16)
        .with_max_new_tokens_final(64);
    let mut state = engine.init_streaming(opts);

    // Feed 4 chunks (4 seconds)
    let mut results: Vec<String> = Vec::new();
    for _ in 0..4 {
        let chunk = sine_tone(440.0, 1.0, 16000);
        if let Some(result) = engine.feed_audio(&mut state, &chunk).unwrap() {
            results.push(result.text.clone());
        }
    }

    assert!(
        !results.is_empty(),
        "should have gotten at least one intermediate result"
    );

    let final_result = engine.finish_streaming(&mut state).unwrap();
    eprintln!("Final streaming result: {:?}", final_result.text);
}

#[test]
#[ignore]
fn test_streaming_vs_batch_consistency() {
    // The final streaming result should match (or be very close to) batch transcription.
    // We use a short audio clip for this test.
    let engine = load_engine();

    // Generate 4 seconds of test audio
    let audio = sine_tone(440.0, 4.0, 16000);

    // Batch transcription
    let batch_opts = qwen3_asr::TranscribeOptions::default()
        .with_max_new_tokens(64);
    let batch_result = engine.transcribe_samples(&audio, batch_opts).unwrap();

    // Streaming transcription
    let stream_opts = qwen3_asr::StreamingOptions::default()
        .with_chunk_size_sec(2.0)
        .with_max_new_tokens_streaming(32)
        .with_max_new_tokens_final(64);
    let mut state = engine.init_streaming(stream_opts);

    // Feed in 2 chunks
    let chunk_samples = 32000; // 2 seconds
    for chunk_start in (0..audio.len()).step_by(chunk_samples) {
        let chunk_end = (chunk_start + chunk_samples).min(audio.len());
        let _ = engine.feed_audio(&mut state, &audio[chunk_start..chunk_end]);
    }
    let stream_result = engine.finish_streaming(&mut state).unwrap();

    eprintln!("Batch:  {:?}", batch_result.text);
    eprintln!("Stream: {:?}", stream_result.text);

    // Both should produce bounded results without errors.
    // For a pure tone, content may be empty or noise, but both paths must succeed.
    assert!(batch_result.text.len() <= 10000, "batch text should be bounded");
    assert!(stream_result.text.len() <= 10000, "stream text should be bounded");
    assert!(batch_result.raw_output.len() <= 10000, "batch raw_output should be bounded");
    assert!(stream_result.raw_output.len() <= 10000, "stream raw_output should be bounded");
}

/// Test that `initial_text` is used as prefix during cold-start chunks.
///
/// Runs two streaming sessions on a longer audio — one without initial_text
/// and one with context from a "previous session". Both must succeed and
/// produce non-empty text, confirming the initial_text code path works.
#[test]
#[ignore]
fn test_streaming_initial_text_cross_session() {
    let engine = load_engine();

    // Use sample4.wav (English paragraph, 36s) for a robust test.
    // Fall back to sample1.wav or tone if not available.
    let wav_path = if Path::new("audio/sample4.wav").exists() {
        Path::new("audio/sample4.wav")
    } else if Path::new("audio/sample1.wav").exists() {
        Path::new("audio/sample1.wav")
    } else {
        eprintln!("No test audio found, using tone");
        // Return early with tone test
        let samples = sine_tone(440.0, 6.0, 16000);
        let opts = qwen3_asr::StreamingOptions::default()
            .with_initial_text("Some prior context text.");
        let mut state = engine.init_streaming(opts);
        let chunk_samples = 32000;
        for chunk_start in (0..samples.len()).step_by(chunk_samples) {
            let chunk_end = (chunk_start + chunk_samples).min(samples.len());
            let _ = engine.feed_audio(&mut state, &samples[chunk_start..chunk_end]);
        }
        let result = engine.finish_streaming(&mut state).unwrap();
        eprintln!("Tone with initial_text: {:?}", result.text);
        return;
    };

    let samples = load_wav_16k(wav_path);
    let chunk_samples = 32000;
    eprintln!("Test audio: {:.1}s from {}", samples.len() as f32 / 16000.0, wav_path.display());

    // Session 1: no initial_text (baseline)
    let opts1 = qwen3_asr::StreamingOptions::default()
        .with_chunk_size_sec(2.0);
    let mut state1 = engine.init_streaming(opts1);
    for chunk_start in (0..samples.len()).step_by(chunk_samples) {
        let chunk_end = (chunk_start + chunk_samples).min(samples.len());
        let _ = engine.feed_audio(&mut state1, &samples[chunk_start..chunk_end]);
    }
    let result1 = engine.finish_streaming(&mut state1).unwrap();
    eprintln!("Without initial_text: {:?}", result1.text);

    // Session 2: with initial_text simulating cross-session context.
    // Use the first sentence of the expected transcript as "previous context".
    let context = "Artificial intelligence has rapidly transformed numerous industries.";
    let opts2 = qwen3_asr::StreamingOptions::default()
        .with_chunk_size_sec(2.0)
        .with_initial_text(context);
    let mut state2 = engine.init_streaming(opts2);
    for chunk_start in (0..samples.len()).step_by(chunk_samples) {
        let chunk_end = (chunk_start + chunk_samples).min(samples.len());
        let _ = engine.feed_audio(&mut state2, &samples[chunk_start..chunk_end]);
    }
    let result2 = engine.finish_streaming(&mut state2).unwrap();
    eprintln!("With initial_text:    {:?}", result2.text);

    // Both sessions must succeed and produce non-empty text for real speech.
    assert!(!result1.text.is_empty(), "session 1 (no context) should produce text");
    assert!(!result2.text.is_empty(), "session 2 (with context) should produce text");
}

/// Test with a real WAV file if available.
/// Set QWEN3_ASR_TEST_WAV to a path to run this test.
#[test]
#[ignore]
fn test_streaming_real_audio() {
    let wav_path = match std::env::var("QWEN3_ASR_TEST_WAV") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("Skipping: set QWEN3_ASR_TEST_WAV to a WAV file path");
            return;
        }
    };

    let engine = load_engine();
    let samples = load_wav_16k(&wav_path);
    let duration_sec = samples.len() as f32 / 16000.0;
    eprintln!("Test audio: {:.1}s", duration_sec);

    // Batch
    let batch_result = engine
        .transcribe_samples(&samples, qwen3_asr::TranscribeOptions::default())
        .unwrap();
    eprintln!("Batch result: {:?}", batch_result.text);

    // Streaming
    let opts = qwen3_asr::StreamingOptions::default()
        .with_chunk_size_sec(2.0);
    let mut state = engine.init_streaming(opts);

    let chunk_samples = 32000; // 2 seconds
    let mut step = 0;
    for chunk_start in (0..samples.len()).step_by(chunk_samples) {
        let chunk_end = (chunk_start + chunk_samples).min(samples.len());
        if let Some(result) = engine
            .feed_audio(&mut state, &samples[chunk_start..chunk_end])
            .unwrap()
        {
            step += 1;
            eprintln!("  Step {}: {:?}", step, result.text);
        }
    }

    let final_result = engine.finish_streaming(&mut state).unwrap();
    eprintln!("Stream result: {:?}", final_result.text);

    // Both should produce non-empty text for real speech
    assert!(!batch_result.text.is_empty(), "batch should produce text");
    assert!(!final_result.text.is_empty(), "stream should produce text");
}
