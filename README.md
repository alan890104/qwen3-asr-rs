# qwen3-asr-rs

Pure-Rust inference engine for **Qwen3-ASR** automatic speech recognition models ([Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)) built on [candle](https://github.com/huggingface/candle). Runs fully locally — no Python, no PyTorch.

## Features

- **All model sizes** — 0.6B and 1.7B work out of the box; select the model directory at runtime
- **Metal GPU acceleration** — Apple Silicon (M1/M2/M3/M4) via candle's Metal backend
- **CUDA support** — enable the `cuda` feature for NVIDIA GPUs
- **Multilingual** — English, Chinese, and code-switched audio (mixed-language)
- **Sharded weights** — loads both single-file and multi-shard `safetensors` models
- **Accurate mel extraction** — matches the official `WhisperFeatureExtractor` (Slaney-normalized, 128 mel bins)
- **BF16 throughout** — matches the reference PyTorch BF16 output exactly
- **MRoPE** — full multi-dimensional rotary position embedding for the Qwen3 decoder
- **No runtime dependencies** — statically linked, single binary

## Demo

Five audio samples are included in [`audio/`](audio/) — two short and three long — covering English, Mandarin, and code-switched speech. Click the links below to play each file directly in your browser (GitHub renders WAV files inline).

### Short Samples — exact match

#### sample1.wav · English · 3 s

[▶ audio/sample1.wav](audio/sample1.wav)

| | Text |
|---|---|
| **Expected** | The quick brown fox jumps over the lazy dog. |
| **Rust output** | The quick brown fox jumps over the lazy dog. |

---

#### sample2.wav · English · 4 s

[▶ audio/sample2.wav](audio/sample2.wav)

| | Text |
|---|---|
| **Expected** | Speech recognition has improved a lot in recent years. |
| **Rust output** | Speech recognition has improved a lot in recent years. |

---

### Long Samples

#### sample4.wav · English paragraph · 36 s

[▶ audio/sample4.wav](audio/sample4.wav)

**Expected:**
> Artificial intelligence has rapidly transformed numerous industries over the past decade. From healthcare diagnostics to autonomous vehicles, machine learning models are now capable of performing tasks that once required years of human expertise. Natural language processing, in particular, has seen dramatic improvements, enabling computers to understand, generate, and translate human speech with remarkable accuracy. Researchers continue to push the boundaries of what is possible, developing systems that can reason, plan, and even demonstrate creativity.

**Rust output (0.6B):**
> Artificial intelligence has rapidly transformed numerous industries over the past decade. From healthcare diagnostics to autonomous vehicles, machine learning models are now capable of performing tasks that once required years of human expertise. Natural language processing, in particular, has seen dramatic improvements, enabling computers to understand, generate, and translate human speech with remarkable accuracy. Researchers continue to push the boundaries of what is possible, developing systems that can reason, plan, and even demonstrate creativity.

---

#### sample5.wav · Mandarin paragraph · 30 s

[▶ audio/sample5.wav](audio/sample5.wav)

**Expected:**
> 随着科技的不断进步，人工智能已经深入到我们日常生活的每个角落。在医疗领域，智能诊断系统能够通过分析医学影像，快速准确地识别疾病。在交通领域，自动驾驶技术正在逐步走向成熟。在教育领域，个性化学习系统能够根据每个学生的学习进度，提供量身定制的教学内容，让每个孩子都能得到最适合自己的教育。

**Rust output (0.6B):**
> 随着科技的不断进步，人工智能已经深入到我们日常生活的每个角落。在医疗领域，智能诊断系统能够通过分析医学影像，快速准确地识别疾病。在交通领域，自动驾驶技术正在逐步走向成熟。在教育领域，个性化学习系统能够根据每个学生的学习进度，提供量身定制的教学内容，让每个孩子都能得到最适合自己的教育。

---

#### sample6.wav · Code-switched (Chinese + English) · 29 s

[▶ audio/sample6.wav](audio/sample6.wav)

**Expected:**
> 今天我们来讨论一下大语言模型的发展现状。Large language models like GPT and Claude have shown impressive results on a wide range of benchmarks, demonstrating strong reasoning and language understanding capabilities. 未来，随着多模态技术的进步，这些模型将能够同时处理文字、图像和语音，实现更加自然和智能的人机交互。

**Rust output (0.6B):**
> 今天我们来讨论一下大语言模型的发展现状。Large language models like GPT and Claude have shown impressive results on a wide range of benchmarks demonstrating strong reasoning and language understanding capabilities. 未来，随着多模态技术的进步，这些模型将能够同时处理文字、图像和语音，实现更加自然和智能的人机交互。

---

## Architecture

Qwen3-ASR combines a Whisper-style audio encoder with a Qwen3 causal language model decoder:

```
Audio → Mel spectrogram (128 bins) → Conv2d ×3 downsampler
      → Transformer encoder (18L / 0.6B, 24L / 1.7B)
      → Linear projection → Qwen3 decoder (28L GQA + MRoPE) → Text
```

## Quick Start

### 1. Download a model

```bash
pip install huggingface_hub

# 0.6B (~3.4 GB)
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir models

# 1.7B (~4.5 GB)
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir models_1.7b
```

### 2. Build and run

```bash
# Apple Silicon (Metal)
cargo run --release

# 1.7B model
cargo run --release -- models_1.7b

# CPU only
cargo run --release --no-default-features
```

### 3. Transcribe your own audio

```bash
MODEL_DIR=models cargo run --release -- path/to/audio.wav
```

> Audio is automatically resampled to 16 kHz mono. WAV, and any format supported by `hound` are accepted.

## Test Results

Tested on Apple M-series with Metal acceleration.

| Sample | Duration | Model | Result |
|--------|----------|-------|--------|
| English sentence | 3 s | 0.6B | ✓ exact match |
| English sentence | 3 s | 1.7B | ✓ exact match |
| Long English paragraph | 45 s | 0.6B | ✓ exact match |
| Long English paragraph | 45 s | 1.7B | ✓ exact match |
| Long Chinese paragraph | 30 s | 0.6B | ✓ exact match |
| Long Chinese paragraph | 30 s | 1.7B | ✓ near-exact match |
| Mixed Chinese-English | 25 s | 0.6B | ✓ full transcription |
| Mixed Chinese-English | 25 s | 1.7B | ✓ full transcription |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `candle-core` / `candle-nn` | Tensor ops, Metal/CUDA backends |
| `tokenizers` | HuggingFace tokenizer (BPE) |
| `hound` | WAV file I/O |
| `rubato` | High-quality audio resampling |
| `rustfft` | FFT for mel spectrogram |
| `safetensors` | Model weight loading |

## Enabling CUDA

```toml
# Cargo.toml
[features]
default = ["cuda"]
cuda = ["candle-core/cuda", "candle-nn/cuda"]
```

## Implementation Notes

- **Mel extraction** matches `WhisperFeatureExtractor` exactly: Slaney-normalized filterbanks, `n_fft=400`, `hop_length=160`, `n_mels=128`, max diff < 3e-5 vs PyTorch reference
- **Positional embeddings** in the audio encoder are sinusoidal and computed per-chunk (positions reset to 0 for each 30-second window), matching the Python reference
- **BF16 precision** is used throughout — LayerNorm and softmax are computed in F32 and cast back — this matches the official PyTorch BF16 output
- **Token 151704** (`<asr_sep>`) splits the decoder output into `language` and `text` fields; it is absent from the base Qwen3 tokenizer (decodes to `""`) so it is detected by token ID directly

## License

MIT
