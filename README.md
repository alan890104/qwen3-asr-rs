# qwen3-asr-rs

[![Crates.io](https://img.shields.io/crates/v/qwen3-asr.svg)](https://crates.io/crates/qwen3-asr)
[![docs.rs](https://docs.rs/qwen3-asr/badge.svg)](https://docs.rs/qwen3-asr)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust **speech-to-text** engine for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) — run automatic speech recognition (ASR) locally with **GPU acceleration** on Metal and CUDA. No Python, no PyTorch, no server required.

Built on [candle](https://github.com/huggingface/candle). Supports batch transcription and real-time streaming. Multilingual: English, Chinese, and mixed-language audio.

## Why qwen3-asr-rs?

- **Local & private** — audio never leaves your machine
- **Fast** — 4x real-time on Apple M4, faster on NVIDIA GPUs
- **Simple API** — `engine.transcribe("audio.wav")` or streaming `feed_audio()` / `finish_streaming()`
- **Single binary** — no runtime dependencies, easy to deploy
- **Cross-platform** — macOS (Metal), Linux/Windows (CUDA), or CPU fallback

## Features

| Feature | Description |
|---------|-------------|
| Batch transcription | Transcribe complete audio files with maximum accuracy |
| Streaming transcription | Real-time speech-to-text with ~2s latency for live subtitles, voice assistants |
| Metal GPU (default) | Apple Silicon M1/M2/M3/M4 acceleration via candle Metal backend |
| CUDA GPU | NVIDIA GPU acceleration via candle CUDA backend |
| Multilingual | English, Chinese, code-switched (mixed-language) audio |
| HuggingFace Hub | Auto-download GGUF quantized models with `hub` feature |
| Multiple model sizes | 0.6B (1.7 GB) and 1.7B (4.5 GB) safetensors models |

## Quick Start

### Install as a dependency

```toml
# Cargo.toml

# macOS (Metal GPU, default)
[dependencies]
qwen3-asr = "0.2"

# Linux / Windows (NVIDIA CUDA GPU)
[dependencies]
qwen3-asr = { version = "0.2", default-features = false, features = ["cuda"] }

# CPU only (any platform)
[dependencies]
qwen3-asr = { version = "0.2", default-features = false }
```

### Transcribe an audio file

```rust
use qwen3_asr::{AsrInference, TranscribeOptions, best_device};

let device = best_device(); // automatically selects CUDA → Metal → CPU
let engine = AsrInference::load("path/to/model", device)?;
let result = engine.transcribe("audio.wav", TranscribeOptions::default())?;

println!("Language: {}", result.language);
println!("Text: {}", result.text);
```

### Real-time streaming from microphone

```rust
use qwen3_asr::StreamingOptions;

let mut state = engine.init_streaming(StreamingOptions::default());

// Feed audio chunks (16 kHz f32) as they arrive
for chunk in mic_chunks {
    if let Some(result) = engine.feed_audio(&mut state, &chunk)? {
        println!("Live: {}", result.text);
    }
}

let final_result = engine.finish_streaming(&mut state)?;
println!("Final: {}", final_result.text);
```

### Download a model

```bash
pip install huggingface_hub

# 0.6B (~1.7 GB) — fast, recommended for real-time
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models

# 1.7B (~4.5 GB) — higher accuracy
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir models_1.7b
```

Or use the `hub` feature to auto-download GGUF quantized models:

```rust
let engine = AsrInference::from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    Path::new("models/"),
    device,
)?;
```

### Build and run the CLI

```bash
# Apple Silicon (Metal)
cargo run --release

# NVIDIA GPU (CUDA)
cargo run --release --no-default-features --features cuda

# CPU only
cargo run --release --no-default-features
```

## Benchmark

Tested on Apple Mac mini M4 (16 GB), Metal backend, mean of 3 runs per sample.

### Real-Time Factor (RTF)

**RTF = inference time / audio duration.** Below 1.0 = faster than real-time. Lower is better.

| Model | 3s English | 4s English | 36s English | 30s Chinese | 29s Mixed | **Avg RTF** |
|-------|-----------|-----------|-------------|-------------|-----------|-------------|
| 0.6B BF16 | 0.149 | 0.136 | 0.254 | 0.237 | 0.216 | **0.230** |
| 1.7B BF16 | 0.307 | 0.253 | 0.338 | 0.324 | 0.302 | **0.319** |

Both models run 3-7x faster than real-time on Apple M4 Metal.

### Model Load Time and Memory

| Model | File Size | Load Time | Memory (live) |
|-------|-----------|-----------|---------------|
| Qwen3-ASR-0.6B | 1.7 GB | 489 ms | 1.9 GB |
| Qwen3-ASR-1.7B | 4.5 GB | 4250 ms | 4.6 GB |

## Demo: Transcription Samples

Five audio samples covering English, Mandarin, and code-switched speech.

<details>
<summary>sample1.wav — English, 3s (click to expand)</summary>

[▶ audio/sample1.wav](audio/sample1.wav)

| | Text |
|---|---|
| **Expected** | The quick brown fox jumps over the lazy dog. |
| **Rust output** | The quick brown fox jumps over the lazy dog. |

</details>

<details>
<summary>sample2.wav — English, 4s</summary>

[▶ audio/sample2.wav](audio/sample2.wav)

| | Text |
|---|---|
| **Expected** | Speech recognition has improved a lot in recent years. |
| **Rust output** | Speech recognition has improved a lot in recent years. |

</details>

<details>
<summary>sample4.wav — English paragraph, 36s</summary>

[▶ audio/sample4.wav](audio/sample4.wav)

**Expected:**
> Artificial intelligence has rapidly transformed numerous industries over the past decade. From healthcare diagnostics to autonomous vehicles, machine learning models are now capable of performing tasks that once required years of human expertise. Natural language processing, in particular, has seen dramatic improvements, enabling computers to understand, generate, and translate human speech with remarkable accuracy. Researchers continue to push the boundaries of what is possible, developing systems that can reason, plan, and even demonstrate creativity.

**Rust output (0.6B):** exact match.

</details>

<details>
<summary>sample5.wav — Mandarin paragraph, 30s</summary>

[▶ audio/sample5.wav](audio/sample5.wav)

**Expected:**
> 随着科技的不断进步，人工智能已经深入到我们日常生活的每个角落。在医疗领域，智能诊断系统能够通过分析医学影像，快速准确地识别疾病。在交通领域，自动驾驶技术正在逐步走向成熟。在教育领域，个性化学习系统能够根据每个学生的学习进度，提供量身定制的教学内容，让每个孩子都能得到最适合自己的教育。

**Rust output (0.6B):** exact match.

</details>

<details>
<summary>sample6.wav — Code-switched Chinese + English, 29s</summary>

[▶ audio/sample6.wav](audio/sample6.wav)

**Expected:**
> 今天我们来讨论一下大语言模型的发展现状。Large language models like GPT and Claude have shown impressive results on a wide range of benchmarks, demonstrating strong reasoning and language understanding capabilities. 未来，随着多模态技术的进步，这些模型将能够同时处理文字、图像和语音，实现更加自然和智能的人机交互。

**Rust output (0.6B):** near-exact match (minor punctuation difference).

</details>

## Architecture

Qwen3-ASR combines a Whisper-style audio encoder with a Qwen3 causal language model decoder:

```
Audio → Mel spectrogram (128 bins) → Conv2d ×3 downsampler
      → Transformer encoder (18L / 0.6B, 24L / 1.7B)
      → Linear projection → Qwen3 decoder (28L GQA + MRoPE) → Text
```

## Device Selection

`best_device()` automatically selects the best backend at compile time:

| Priority | Feature | Backend | Platform |
|----------|---------|---------|----------|
| 1 | `cuda` | NVIDIA CUDA | Linux / Windows |
| 2 | `metal` (default) | Apple Metal | macOS Apple Silicon |
| 3 | *(fallback)* | CPU | Any |

You can also construct a `candle_core::Device` manually and pass it to `load()` or `from_pretrained()` for full control.

## Streaming Transcription Guide

The streaming API enables low-latency, real-time speech-to-text for live subtitles, voice assistants, and meeting captioning.

### How it works

The streaming algorithm follows Qwen3-ASR's chunked AED with prefix conditioning:

1. Audio accumulates in an internal buffer; every `chunk_size_sec` (default 2s) triggers inference
2. The encoder uses windowed attention (104 tokens / ~16s window) — completed windows are cached, only the current partial window is recomputed
3. The decoder generates text conditioned on a **prefix** built from the previous output minus the last `unfixed_token_num` tokens (rollback strategy)
4. For the first `unfixed_chunk_num` chunks (cold start), no prefix is used

### Memory and duration limits

Each streaming step re-encodes all accumulated audio (mel extraction + decoder prefill). This is consistent with the [official Python implementation](https://github.com/QwenLM/Qwen3-ASR). Costs grow with session duration:

| Duration | Audio tokens | Per-step latency | Practical |
|----------|-------------|------------------|-----------|
| < 2 min | ~780 | fast | smooth |
| 10 min | ~3,900 | ~1s/step | acceptable |
| 20 min | ~7,800 | ~3-5s/step | upper limit |
| 60 min | ~23,400 | 10s+/step | not feasible |

The [Qwen3-ASR technical report](https://arxiv.org/abs/2601.21337) states the model supports "single speech no longer than 20 minutes."

### Recommended patterns for production

**Short sessions (< 20 min)** — use the streaming API directly.

**Long-running streams (meetings, lectures)** — reset sessions at silence boundaries. Pass `initial_text` from the previous session for cross-session context continuity. Starting a new session is instant (zero-cost struct init, no model reload):

```rust
let make_opts = |ctx: Option<String>| {
    let mut opts = StreamingOptions::default();
    if let Some(text) = ctx {
        opts = opts.with_initial_text(text);
    }
    opts
};
let mut state = engine.init_streaming(make_opts(None));

loop {
    let chunk = read_mic();
    if vad_detects_silence(&chunk, threshold) {
        let result = engine.finish_streaming(&mut state)?;
        save_transcript(&result);
        // Pass last ~200 chars as context for the next session
        let ctx = result.text.chars().rev().take(200).collect::<String>()
            .chars().rev().collect::<String>();
        state = engine.init_streaming(make_opts(Some(ctx)));
    } else {
        if let Some(result) = engine.feed_audio(&mut state, &chunk)? {
            display_subtitle(&result.text);
        }
    }
}
```

Each session stays short (one utterance), so memory and latency remain constant. `initial_text` provides vocabulary/style continuity across resets. This pattern runs indefinitely.

**Long pre-recorded files (podcasts, recordings)** — do not use streaming. Split audio into segments and batch-transcribe each:

```rust
let segments = split_at_silence(&audio); // your VAD / energy-based splitter
for seg in &segments {
    let result = engine.transcribe_samples(seg, TranscribeOptions::default())?;
    output.push(result.text);
}
let full_text = output.join("");
```

This matches the [official approach](https://github.com/QwenLM/Qwen3-ASR) — their `transcribe()` splits long audio at low-energy boundaries (max 1200s/segment) and processes segments independently.

## Dependencies

| Crate | Purpose |
|-------|---------|
| [`candle-core`](https://crates.io/crates/candle-core) / [`candle-nn`](https://crates.io/crates/candle-nn) | Tensor ops, Metal/CUDA backends |
| [`tokenizers`](https://crates.io/crates/tokenizers) | HuggingFace tokenizer (BPE) |
| [`hound`](https://crates.io/crates/hound) | WAV file I/O |
| [`rubato`](https://crates.io/crates/rubato) | Audio resampling to 16 kHz |
| [`rustfft`](https://crates.io/crates/rustfft) | FFT for mel spectrogram |
| [`safetensors`](https://crates.io/crates/safetensors) | Model weight loading |

## Contributing

Contributions are welcome! Please open an issue or pull request on [GitHub](https://github.com/alan890104/qwen3-asr-rs).

## License

MIT
