#!/usr/bin/env python3
"""
Python inference verification using a GGUF Qwen3-ASR file.

Usage:
    pip install gguf numpy tokenizers soundfile scipy
    python tools/infer_gguf.py <model.gguf> <audio.wav> [--language zh]

Purpose:
    Verifies that the GGUF file contains valid, correctly-quantized weights
    by running a full forward pass and printing the transcription.
"""

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFReader, GGMLQuantizationType
except ImportError:
    print("ERROR: gguf not installed. Run: pip install gguf", file=sys.stderr)
    sys.exit(1)

try:
    import tokenizers as hf_tokenizers
except ImportError:
    print("ERROR: tokenizers not installed. Run: pip install tokenizers", file=sys.stderr)
    sys.exit(1)

try:
    import soundfile as sf
    import scipy.signal as ssig
except ImportError:
    print("ERROR: soundfile/scipy not installed. Run: pip install soundfile scipy", file=sys.stderr)
    sys.exit(1)

# ─── GGUF dequantization ──────────────────────────────────────────────────────

DTYPE_Q8_0 = np.dtype([("d", np.float16), ("qs", np.int8, (32,))])


def dequantize_q8_0(raw: np.ndarray, shape: tuple) -> np.ndarray:
    data = raw.view(DTYPE_Q8_0)
    scale = data["d"].astype(np.float32)          # (n_blocks,)
    quants = data["qs"].astype(np.float32)         # (n_blocks, 32)
    flat = (quants * scale[:, None]).reshape(-1)
    n_elements = math.prod(shape)
    return flat[:n_elements].reshape(shape).astype(np.float32)


def load_gguf_tensors(reader: GGUFReader) -> dict:
    """Return {name: np.ndarray float32} for all tensors."""
    tensors = {}
    for t in reader.tensors:
        name = t.name
        shape = tuple(reversed(t.shape.tolist()))  # GGUF stores shape reversed
        dtype = t.tensor_type
        data = t.data  # raw bytes as np.ndarray uint8

        if dtype == GGMLQuantizationType.Q8_0:
            arr = dequantize_q8_0(data, shape)
        elif dtype == GGMLQuantizationType.F16:
            arr = data.view(np.float16).reshape(shape).astype(np.float32)
        elif dtype == GGMLQuantizationType.F32:
            arr = data.view(np.float32).reshape(shape)
        elif dtype == GGMLQuantizationType.BF16:
            raw16 = data.view(np.uint16).reshape(shape)
            arr = (raw16.astype(np.uint32) << 16).view(np.float32)
        else:
            print(f"  WARNING: unsupported dtype {dtype} for {name}, skipping", file=sys.stderr)
            continue
        tensors[name] = arr
    return tensors


def load_gguf_metadata(reader: GGUFReader) -> dict:
    meta = {}
    for key, field in reader.fields.items():
        parts = field.parts
        if len(parts) == 0:
            continue
        # Last part is usually the value(s)
        last = parts[-1]
        if last.dtype.kind in ("U", "S") or last.dtype == object:
            meta[key] = str(last[0])
        elif last.dtype == np.bool_:
            meta[key] = bool(last[0])
        elif np.issubdtype(last.dtype, np.integer):
            vals = last.tolist()
            meta[key] = vals[0] if len(vals) == 1 else vals
        elif np.issubdtype(last.dtype, np.floating):
            vals = last.tolist()
            meta[key] = vals[0] if len(vals) == 1 else vals
        else:
            meta[key] = last.tolist()
    return meta


# ─── Mel spectrogram ──────────────────────────────────────────────────────────

TARGET_SR = 16000
N_FFT = 400
HOP_LENGTH = 160


def load_audio(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n_out = int(len(audio) * target_sr / sr)
        audio = ssig.resample(audio, n_out)
    return audio


def hanning(n: int) -> np.ndarray:
    return np.hanning(n + 1)[:-1]


def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Compute mel filterbank matrix (n_mels, n_fft//2+1)."""
    fmax = sr / 2.0

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = mel_to_hz(mels)
    freq_bins = np.linspace(0, fmax, n_fft // 2 + 1)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, center, hi = freqs[i], freqs[i + 1], freqs[i + 2]
        for j, f in enumerate(freq_bins):
            if lo <= f <= center:
                fb[i, j] = (f - lo) / (center - lo + 1e-8)
            elif center < f <= hi:
                fb[i, j] = (hi - f) / (hi - center + 1e-8)
    return fb


def extract_mel(samples: np.ndarray, n_mels: int = 128) -> np.ndarray:
    """Return mel spectrogram (n_mels, n_frames) in log scale."""
    window = hanning(N_FFT).astype(np.float32)
    n_frames = 1 + (len(samples) - N_FFT) // HOP_LENGTH
    if n_frames <= 0:
        # Pad to at least one frame
        samples = np.pad(samples, (0, N_FFT))
        n_frames = 1

    # STFT
    spec = np.zeros((N_FFT // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        frame = samples[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] * window
        spec[:, i] = np.fft.rfft(frame, n=N_FFT)

    power = np.abs(spec) ** 2  # (n_fft//2+1, n_frames)
    fb = mel_filterbank(TARGET_SR, N_FFT, n_mels)  # (n_mels, n_fft//2+1)
    mel = fb @ power  # (n_mels, n_frames)
    mel = np.log(mel + 1e-10)
    return mel.astype(np.float32)


# ─── Numpy forward pass ───────────────────────────────────────────────────────

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return w * (x - mean) / np.sqrt(var + eps) + b


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return w * x / rms


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    out = x @ w.T
    if b is not None:
        out = out + b
    return out


def conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None, stride: int = 1, padding: int = 0) -> np.ndarray:
    """x: (L, C_in), w: (C_out, C_in, K)"""
    L, C_in = x.shape
    C_out, C_in_w, K = w.shape
    assert C_in == C_in_w
    if padding > 0:
        x = np.pad(x, ((padding, padding), (0, 0)))
    L_pad = x.shape[0]
    L_out = (L_pad - K) // stride + 1
    out = np.zeros((L_out, C_out), dtype=np.float32)
    for k in range(K):
        out += x[k : k + L_out * stride : stride, :] @ w[:, :, k].T
    if b is not None:
        out += b
    return out


def conv2d_4d(x: np.ndarray, w: np.ndarray, b: np.ndarray | None, stride=(1, 1)) -> np.ndarray:
    """
    x: (C_in, H, W), w: (C_out, C_in, kH, kW)
    Returns (C_out, H_out, W_out)
    """
    C_in, H, W = x.shape
    C_out, C_in_w, kH, kW = w.shape
    sH, sW = stride
    H_out = (H - kH) // sH + 1
    W_out = (W - kW) // sW + 1
    out = np.zeros((C_out, H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, i * sH : i * sH + kH, j * sW : j * sW + kW]  # (C_in, kH, kW)
            out[:, i, j] = (w.reshape(C_out, -1) @ patch.reshape(-1)) + (b if b is not None else 0)
    return out


# ─── Audio Encoder ────────────────────────────────────────────────────────────

class AudioEncoder:
    def __init__(self, W: dict, prefix: str, config: dict):
        self.W = W
        self.p = prefix
        self.config = config
        self.d_model = config["d_model"]
        self.n_heads = config["encoder_attention_heads"]
        self.head_dim = self.d_model // self.n_heads
        self.n_layers = config["encoder_layers"]
        self.n_mels = config["num_mel_bins"]
        self.output_dim = config["output_dim"]

    def g(self, name: str) -> np.ndarray:
        key = f"{self.p}.{name}"
        if key not in self.W:
            raise KeyError(f"Weight not found: {key}")
        return self.W[key]

    def forward(self, mel: np.ndarray) -> np.ndarray:
        """mel: (n_mels, n_frames) -> (seq_len, output_dim)"""
        # conv_layers: list of Conv2d
        # Qwen3-ASR audio encoder uses Conv2D on mel as image (1, n_mels, n_frames)
        x = mel[np.newaxis, :, :]  # (1, n_mels, n_frames)

        # Conv1: (d_model, 1, 3, 1) stride (2, 1)
        w0 = self.g("conv1.weight")  # (d_model, 1, kH, kW)
        b0 = self.g("conv1.bias") if f"{self.p}.conv1.bias" in self.W else None
        # Apply conv2d
        x = conv2d_4d(x, w0, b0, stride=(2, 1))  # (d_model, H', W)
        x = gelu(x)

        w1 = self.g("conv2.weight")
        b1 = self.g("conv2.bias") if f"{self.p}.conv2.bias" in self.W else None
        x = conv2d_4d(x, w1, b1, stride=(2, 1))
        x = gelu(x)

        # Reshape: (C, H', W) -> (W, C*H') or handle as (frames, features)
        C, H_out, W = x.shape
        x = x.transpose(2, 0, 1).reshape(W, C * H_out)  # (n_frames_out, features)

        # Project to d_model
        proj_w = self.g("linear_in.weight")
        proj_b_key = f"{self.p}.linear_in.bias"
        proj_b = self.W.get(proj_b_key)
        x = linear(x, proj_w, proj_b)  # (n_frames_out, d_model)

        # Positional embedding (sinusoidal or learned)
        # Apply transformer encoder layers
        for layer_idx in range(self.n_layers):
            lp = f"layers.{layer_idx}"
            # Self-attention
            ln1_w = self.g(f"{lp}.self_attn_layer_norm.weight")
            ln1_b = self.g(f"{lp}.self_attn_layer_norm.bias")
            residual = x
            x_ln = layer_norm(x, ln1_w, ln1_b)

            # Attention projections
            q = linear(x_ln, self.g(f"{lp}.self_attn.q_proj.weight"),
                       self.W.get(f"{self.p}.{lp}.self_attn.q_proj.bias"))
            k = linear(x_ln, self.g(f"{lp}.self_attn.k_proj.weight"),
                       self.W.get(f"{self.p}.{lp}.self_attn.k_proj.bias"))
            v = linear(x_ln, self.g(f"{lp}.self_attn.v_proj.weight"),
                       self.W.get(f"{self.p}.{lp}.self_attn.v_proj.bias"))

            T, _ = x_ln.shape
            q = q.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)
            k = k.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)
            v = v.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)

            scale = math.sqrt(self.head_dim)
            attn = softmax((q @ k.transpose(0, 2, 1)) / scale)
            out = (attn @ v).transpose(1, 0, 2).reshape(T, self.d_model)
            out = linear(out, self.g(f"{lp}.self_attn.out_proj.weight"),
                         self.W.get(f"{self.p}.{lp}.self_attn.out_proj.bias"))
            x = residual + out

            # FFN
            ln2_w = self.g(f"{lp}.final_layer_norm.weight")
            ln2_b = self.g(f"{lp}.final_layer_norm.bias")
            residual = x
            x_ln = layer_norm(x, ln2_w, ln2_b)
            fc1_w = self.g(f"{lp}.fc1.weight")
            fc1_b = self.W.get(f"{self.p}.{lp}.fc1.bias")
            fc2_w = self.g(f"{lp}.fc2.weight")
            fc2_b = self.W.get(f"{self.p}.{lp}.fc2.bias")
            x_ff = gelu(linear(x_ln, fc1_w, fc1_b))
            x_ff = linear(x_ff, fc2_w, fc2_b)
            x = residual + x_ff

        # Final layer norm
        ln_w = self.g("layer_norm.weight")
        ln_b = self.g("layer_norm.bias")
        x = layer_norm(x, ln_w, ln_b)

        # Project to output_dim
        proj_out_w = self.g("linear_out.weight")
        proj_out_b = self.W.get(f"{self.p}.linear_out.bias")
        x = linear(x, proj_out_w, proj_out_b)

        return x  # (seq_len, output_dim)


# ─── Prompt building ──────────────────────────────────────────────────────────

AUDIO_PAD_TOKEN_ID = 151676
IM_END = 151645
IM_START = 151644
ENDOFTEXT = 151643
ASR_SEP = 151704


def build_prompt_ids(num_audio_tokens: int, language: str | None) -> tuple[list[int], int]:
    tokens = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
    audio_start = len(tokens)
    tokens.extend([AUDIO_PAD_TOKEN_ID] * num_audio_tokens)
    tokens.extend([151670, 151645, 198, 151644])
    if language:
        tokens.extend([77091, 198])
        # language tokens encoded below
    else:
        tokens.extend([77091, 198])
    return tokens, audio_start


# ─── Simple text decoder (numpy) ─────────────────────────────────────────────

class TextDecoder:
    def __init__(self, W: dict, prefix: str, config: dict):
        self.W = W
        self.p = prefix
        self.config = config
        self.n_layers = config["num_hidden_layers"]
        self.n_heads = config["num_attention_heads"]
        self.n_kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.hidden = config["hidden_size"]
        self.intermediate = config["intermediate_size"]
        self.eps = config["rms_norm_eps"]
        self.vocab_size = config["vocab_size"]
        self.rope_theta = config["rope_theta"]
        # KV cache
        self.kv_cache = [(np.zeros((0, self.n_kv_heads, self.head_dim), dtype=np.float32),
                          np.zeros((0, self.n_kv_heads, self.head_dim), dtype=np.float32))
                         for _ in range(self.n_layers)]

    def g(self, name: str) -> np.ndarray:
        key = f"{self.p}.{name}"
        if key not in self.W:
            raise KeyError(f"Weight not found: {key}")
        return self.W[key]

    def reset_cache(self):
        self.kv_cache = [(np.zeros((0, self.n_kv_heads, self.head_dim), dtype=np.float32),
                          np.zeros((0, self.n_kv_heads, self.head_dim), dtype=np.float32))
                         for _ in range(self.n_layers)]

    def rope(self, x: np.ndarray, positions: list[int]) -> np.ndarray:
        """Apply RoPE. x: (T, n_heads, head_dim)"""
        T, H, D = x.shape
        half = D // 2
        freqs = 1.0 / (self.rope_theta ** (np.arange(0, D, 2, dtype=np.float32) / D))
        for t, pos in enumerate(positions):
            angles = pos * freqs  # (half,)
            cos_a = np.cos(angles)
            sin_a = np.sin(angles)
            xr = x[t, :, :half]
            xi = x[t, :, half:]
            x[t, :, :half] = xr * cos_a - xi * sin_a
            x[t, :, half:] = xr * sin_a + xi * cos_a
        return x

    def embed(self, ids: list[int]) -> np.ndarray:
        embed_w = self.g("embed_tokens.weight")
        return embed_w[ids]  # (T, hidden)

    def forward(self, hidden: np.ndarray, positions: list[int]) -> np.ndarray:
        """
        hidden: (T, hidden_size) — new tokens only
        positions: list of int positions for each token
        Returns logits: (T, vocab_size)
        """
        x = hidden.copy()
        for layer_idx in range(self.n_layers):
            lp = f"layers.{layer_idx}"
            # Input RMSNorm
            ln_w = self.g(f"{lp}.input_layernorm.weight")
            residual = x
            x_ln = rms_norm(x, ln_w, self.eps)

            # QKV
            T_new = x_ln.shape[0]
            qw = self.g(f"{lp}.self_attn.q_proj.weight")
            kw = self.g(f"{lp}.self_attn.k_proj.weight")
            vw = self.g(f"{lp}.self_attn.v_proj.weight")

            # QKV norms (Qwen3 uses per-head QK norm)
            q = (x_ln @ qw.T).reshape(T_new, self.n_heads, self.head_dim)
            k = (x_ln @ kw.T).reshape(T_new, self.n_kv_heads, self.head_dim)
            v = (x_ln @ vw.T).reshape(T_new, self.n_kv_heads, self.head_dim)

            # Per-head QK norm
            qnorm_key = f"{self.p}.{lp}.self_attn.q_norm.weight"
            knorm_key = f"{self.p}.{lp}.self_attn.k_norm.weight"
            if qnorm_key in self.W:
                qnw = self.W[qnorm_key]
                q = rms_norm(q, qnw, self.eps)
            if knorm_key in self.W:
                knw = self.W[knorm_key]
                k = rms_norm(k, knw, self.eps)

            # RoPE
            q = self.rope(q, positions)
            k = self.rope(k, positions)

            # KV cache
            past_k, past_v = self.kv_cache[layer_idx]
            k_full = np.concatenate([past_k, k], axis=0)
            v_full = np.concatenate([past_v, v], axis=0)
            self.kv_cache[layer_idx] = (k_full, v_full)

            T_full = k_full.shape[0]

            # GQA: repeat KV heads
            reps = self.n_heads // self.n_kv_heads
            k_rep = np.repeat(k_full, reps, axis=1)  # (T_full, n_heads, head_dim)
            v_rep = np.repeat(v_full, reps, axis=1)

            # Attention
            q_t = q.transpose(1, 0, 2)        # (n_heads, T_new, head_dim)
            k_t = k_rep.transpose(1, 2, 0)    # (n_heads, head_dim, T_full)
            v_t = v_rep.transpose(1, 0, 2)    # (n_heads, T_full, head_dim)

            scale = math.sqrt(self.head_dim)
            scores = (q_t @ k_t) / scale      # (n_heads, T_new, T_full)

            # Causal mask for new tokens
            past_len = T_full - T_new
            if T_new > 1:
                mask = np.triu(np.full((T_new, T_full), -1e9), k=past_len + 1)
                scores += mask

            attn = softmax(scores, axis=-1)
            attn_out = (attn @ v_t).transpose(1, 0, 2).reshape(T_new, self.n_heads * self.head_dim)

            # Output proj
            ow = self.g(f"{lp}.self_attn.o_proj.weight")
            attn_out = attn_out @ ow.T
            x = residual + attn_out

            # Post-attention norm + FFN
            post_ln_w = self.g(f"{lp}.post_attention_layernorm.weight")
            residual = x
            x_ln = rms_norm(x, post_ln_w, self.eps)

            gate_w = self.g(f"{lp}.mlp.gate_proj.weight")
            up_w = self.g(f"{lp}.mlp.up_proj.weight")
            down_w = self.g(f"{lp}.mlp.down_proj.weight")

            gate = silu(x_ln @ gate_w.T)
            up = x_ln @ up_w.T
            x_ff = (gate * up) @ down_w.T
            x = residual + x_ff

        # Final norm
        norm_w = self.g("norm.weight")
        x = rms_norm(x, norm_w, self.eps)

        # LM head
        lm_head_key = "thinker.lm_head.weight"
        if lm_head_key in self.W:
            lm_head_w = self.W[lm_head_key]
        else:
            lm_head_w = self.g("embed_tokens.weight")  # tied
        logits = x @ lm_head_w.T  # (T, vocab)
        return logits


# ─── Main inference loop ──────────────────────────────────────────────────────

def transcribe(gguf_path: str, audio_path: str, language: str | None = None) -> str:
    print(f"Loading GGUF: {gguf_path}", flush=True)
    reader = GGUFReader(gguf_path)
    meta = load_gguf_metadata(reader)

    # Extract config
    audio_cfg = {
        "d_model":                int(meta.get("qwen3_asr.audio.d_model", 896)),
        "encoder_layers":         int(meta.get("qwen3_asr.audio.encoder_layers", 18)),
        "encoder_attention_heads":int(meta.get("qwen3_asr.audio.encoder_attention_heads", 14)),
        "encoder_ffn_dim":        int(meta.get("qwen3_asr.audio.encoder_ffn_dim", 3584)),
        "num_mel_bins":           int(meta.get("qwen3_asr.audio.num_mel_bins", 128)),
        "max_source_positions":   int(meta.get("qwen3_asr.audio.max_source_positions", 1500)),
        "n_window":               int(meta.get("qwen3_asr.audio.n_window", 50)),
        "n_window_infer":         int(meta.get("qwen3_asr.audio.n_window_infer", 800)),
        "conv_chunksize":         int(meta.get("qwen3_asr.audio.conv_chunksize", 500)),
        "output_dim":             int(meta.get("qwen3_asr.audio.output_dim", 1024)),
    }
    text_cfg = {
        "vocab_size":          int(meta.get("qwen3_asr.text.vocab_size", 151936)),
        "hidden_size":         int(meta.get("qwen3_asr.text.hidden_size", 1024)),
        "intermediate_size":   int(meta.get("qwen3_asr.text.intermediate_size", 3072)),
        "num_hidden_layers":   int(meta.get("qwen3_asr.text.num_hidden_layers", 28)),
        "num_attention_heads": int(meta.get("qwen3_asr.text.num_attention_heads", 16)),
        "num_key_value_heads": int(meta.get("qwen3_asr.text.num_key_value_heads", 8)),
        "head_dim":            int(meta.get("qwen3_asr.text.head_dim", 128)),
        "rms_norm_eps":        float(meta.get("qwen3_asr.text.rms_norm_eps", 1e-6)),
        "rope_theta":          float(meta.get("qwen3_asr.text.rope_theta", 1_000_000.0)),
    }
    n_mels = audio_cfg["num_mel_bins"]

    # Extract tokenizer
    tok_json_str = meta.get("tokenizer.huggingface.json")
    if not tok_json_str:
        print("ERROR: tokenizer.huggingface.json not found in GGUF metadata", file=sys.stderr)
        sys.exit(1)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        f.write(tok_json_str)
        tok_path = f.name

    tokenizer = hf_tokenizers.Tokenizer.from_file(tok_path)

    # Load weights
    print("Dequantizing tensors ...", flush=True)
    W = load_gguf_tensors(reader)
    print(f"Loaded {len(W)} tensors", flush=True)

    # Load audio
    print(f"Loading audio: {audio_path}", flush=True)
    samples = load_audio(audio_path)
    print(f"Audio: {len(samples)} samples @ {TARGET_SR} Hz", flush=True)

    # Mel
    mel = extract_mel(samples, n_mels)
    print(f"Mel: {mel.shape}", flush=True)

    # Audio encoder
    audio_encoder = AudioEncoder(W, "thinker.audio_tower", audio_cfg)
    audio_embeds = audio_encoder.forward(mel)  # (seq_len, output_dim)
    num_audio_tokens = audio_embeds.shape[0]
    print(f"Audio tokens: {num_audio_tokens}", flush=True)

    # Text decoder
    text_decoder = TextDecoder(W, "thinker.model", text_cfg)
    text_decoder.reset_cache()

    # Build prompt
    prompt_ids, audio_start = build_prompt_ids(num_audio_tokens, language)

    if language:
        # Encode language prefix and extend prompt
        lang_cap = language.capitalize()
        lang_enc = tokenizer.encode(f"language {lang_cap}", add_special_tokens=False)
        prompt_ids.extend(lang_enc.ids)

    seq_len = len(prompt_ids)

    # Build embeddings
    before_ids = prompt_ids[:audio_start]
    after_ids = prompt_ids[audio_start + num_audio_tokens:]

    embed_w = W["thinker.model.embed_tokens.weight"]
    before_emb = embed_w[before_ids]                  # (audio_start, hidden)
    audio_emb = audio_embeds                           # (num_audio_tokens, output_dim)
    after_emb = embed_w[after_ids]

    # Pad audio embeddings to hidden_size if needed
    hidden_size = text_cfg["hidden_size"]
    if audio_emb.shape[1] != hidden_size:
        # Use a linear projection (thinker.audio_projection or similar)
        proj_key = "thinker.audio_projection.weight"
        if proj_key in W:
            audio_emb = audio_emb @ W[proj_key].T
        else:
            # Truncate/pad
            if audio_emb.shape[1] > hidden_size:
                audio_emb = audio_emb[:, :hidden_size]
            else:
                audio_emb = np.pad(audio_emb, ((0, 0), (0, hidden_size - audio_emb.shape[1])))

    hidden = np.concatenate([before_emb, audio_emb, after_emb], axis=0)  # (seq_len, hidden)

    # Prefill
    positions = list(range(seq_len))
    logits = text_decoder.forward(hidden, positions)  # (seq_len, vocab)
    next_logits = logits[-1]  # (vocab,)

    # Autoregressive generation
    generated = []
    max_new = 512
    eos_ids = {ENDOFTEXT, IM_END}
    current_pos = seq_len

    for _ in range(max_new):
        next_token = int(np.argmax(next_logits))
        if next_token in eos_ids:
            break
        generated.append(next_token)

        next_emb = embed_w[[next_token]]  # (1, hidden)
        step_logits = text_decoder.forward(next_emb, [current_pos])
        next_logits = step_logits[-1]
        current_pos += 1

    print(f"Generated {len(generated)} tokens", flush=True)
    text = tokenizer.decode(generated)

    # Parse language and text
    if ASR_SEP in generated:
        sep_pos = generated.index(ASR_SEP)
        lang_str = tokenizer.decode(generated[:sep_pos]).strip()
        text = tokenizer.decode(generated[sep_pos + 1:]).strip()
        lang = lang_str.removeprefix("language ").strip()
    elif language:
        lang = language
    else:
        lang = "unknown"

    print(f"Language: {lang}", flush=True)
    print(f"Transcription: {text}", flush=True)
    return text


def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR GGUF inference (Python verification)")
    parser.add_argument("gguf", help="Path to .gguf file")
    parser.add_argument("audio", help="Path to audio .wav file")
    parser.add_argument("--language", "-l", default=None, help="Force language (e.g. zh, en)")
    args = parser.parse_args()

    transcribe(args.gguf, args.audio, args.language)


if __name__ == "__main__":
    main()
