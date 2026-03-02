#!/usr/bin/env python3
"""
Convert Qwen3-ASR safetensors weights to GGUF Q8_0 format.

Usage:
    pip install gguf safetensors numpy
    python tools/convert_to_gguf.py <model_dir> <output.gguf>

Examples:
    python tools/convert_to_gguf.py models/ qwen3_asr_0.6b_q8_0.gguf
    python tools/convert_to_gguf.py models_1.7b/ qwen3_asr_1.7b_q8_0.gguf
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("ERROR: gguf not installed. Run: pip install gguf", file=sys.stderr)
    sys.exit(1)

try:
    from safetensors import safe_open  # noqa: F401 — kept for optional use
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors", file=sys.stderr)
    sys.exit(1)

# Q8_0 block dtype: fp16 scale + 32 x int8 quants = 34 bytes per block
DTYPE_Q8_0 = np.dtype([("d", np.float16), ("qs", np.int8, (32,))])


def to_q8_0(data: np.ndarray) -> np.ndarray:
    """Quantize a float array to Q8_0 block format."""
    flat = data.flatten().astype(np.float32)
    n_blocks = (len(flat) + 31) // 32
    padded = np.zeros(n_blocks * 32, dtype=np.float32)
    padded[: len(flat)] = flat
    blocks = padded.reshape(n_blocks, 32)
    max_abs = np.max(np.abs(blocks), axis=1)
    scale = (max_abs / 127.0).astype(np.float16)
    scale_f32 = scale.astype(np.float32)
    safe_scale = np.where(scale_f32 > 0, scale_f32, 1.0)
    quants = np.round(blocks / safe_scale[:, None]).clip(-127, 127).astype(np.int8)
    result = np.empty(n_blocks, dtype=DTYPE_Q8_0)
    result["d"] = scale
    result["qs"] = quants
    return result


def load_config(model_dir: Path) -> dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        return json.load(f)


DTYPE_MAP = {
    "F32":  np.float32,
    "F16":  np.float16,
    "I8":   np.int8,
    "U8":   np.uint8,
    "I16":  np.int16,
    "I32":  np.int32,
    "I64":  np.int64,
    "U16":  np.uint16,
    "U32":  np.uint32,
    "U64":  np.uint64,
    "BOOL": np.bool_,
}


def _load_st_file(path: Path) -> dict:
    """Low-level safetensors reader that handles BF16 by converting to float32."""
    import struct as _struct
    tensors = {}
    with open(path, "rb") as f:
        header_len = _struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
        for key, info in header.items():
            if key == "__metadata__":
                continue
            dtype_str = info["dtype"]
            shape = tuple(info["shape"])
            start, end = info["data_offsets"]
            f.seek(data_start + start)
            raw = f.read(end - start)
            if dtype_str == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).copy()
                tensors[key] = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            elif dtype_str in DTYPE_MAP:
                tensors[key] = np.frombuffer(raw, dtype=DTYPE_MAP[dtype_str]).copy().reshape(shape)
            else:
                raise ValueError(f"Unsupported dtype '{dtype_str}' for tensor '{key}'")
    return tensors


def load_safetensors(model_dir: Path) -> dict:
    """Load all tensors from safetensors (single file or sharded)."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        tensors = {}
        for shard in shard_files:
            print(f"  Loading shard: {shard}", flush=True)
            tensors.update(_load_st_file(model_dir / shard))
        return tensors

    model_path = model_dir / "model.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    print(f"  Loading: model.safetensors", flush=True)
    return _load_st_file(model_path)


def classify_tensor(name: str, shape: tuple) -> str:
    """Return storage type: 'q8_0', 'bf16', or 'f32'.

    Target dtype policy matches the original BF16 safetensors model:
      - 2D linear weights         → Q8_0  (quantized, dequantize to BF16 at runtime)
      - 4D conv weights           → BF16  (too small to gain from quantization)
      - embed_tokens / lm_head    → BF16  (same as original)
      - 1D norms / biases         → F32   (numerical stability)
    Using BF16 everywhere (instead of F16 for conv/embed) ensures uniform dtype
    flow through the model and avoids F16/BF16 mismatches at residual connections.
    """
    ndim = len(shape)

    # 1D (norms, biases) -> F32 for numerical stability
    if ndim == 1:
        return "f32"

    # 2D weight matrices -> Q8_0 (quantized)
    if ndim == 2:
        return "q8_0"

    # 4D conv weights and anything else -> BF16 (matches original model dtype)
    return "bf16"


def get_model_name(model_dir: Path, config: dict) -> str:
    """Determine a human-readable model name."""
    dir_name = model_dir.resolve().name.lower()
    if "1.7b" in dir_name or "1_7b" in dir_name:
        return "Qwen3-ASR-1.7B"
    # Use hidden_size to distinguish: 0.6B=1024, 1.7B>1024
    hidden_size = (
        config.get("thinker_config", {})
        .get("text_config", {})
        .get("hidden_size", 1024)
    )
    if hidden_size > 1024:
        return "Qwen3-ASR-1.7B"
    return "Qwen3-ASR-0.6B"


def write_gguf(model_dir: Path, output_path: Path) -> None:
    print(f"Loading config from {model_dir} ...", flush=True)
    config = load_config(model_dir)

    tc = config.get("thinker_config", {})
    ac = tc.get("audio_config", {})
    xt = tc.get("text_config", {})

    # Read tokenizer.json as a raw string
    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
    with open(tokenizer_path, encoding="utf-8") as f:
        tokenizer_json_str = f.read()

    model_name = get_model_name(model_dir, config)
    print(f"Model: {model_name}", flush=True)

    print(f"Loading safetensors from {model_dir} ...", flush=True)
    tensors = load_safetensors(model_dir)
    print(f"Loaded {len(tensors)} tensors", flush=True)

    # MRoPE section
    rope_scaling = xt.get("rope_scaling", {}) or {}
    mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    print(f"Creating GGUF writer -> {output_path}", flush=True)
    writer = GGUFWriter(str(output_path), arch="qwen3_asr")

    # ── Metadata ────────────────────────────────────────────────────────────────
    writer.add_name(model_name)

    # Audio encoder config
    writer.add_uint32("qwen3_asr.audio.d_model",                 int(ac.get("d_model", 896)))
    writer.add_uint32("qwen3_asr.audio.encoder_layers",          int(ac.get("encoder_layers", 18)))
    writer.add_uint32("qwen3_asr.audio.encoder_attention_heads", int(ac.get("encoder_attention_heads", 14)))
    writer.add_uint32("qwen3_asr.audio.encoder_ffn_dim",         int(ac.get("encoder_ffn_dim", 3584)))
    writer.add_uint32("qwen3_asr.audio.num_mel_bins",            int(ac.get("num_mel_bins", 128)))
    writer.add_uint32("qwen3_asr.audio.max_source_positions",    int(ac.get("max_source_positions", 1500)))
    writer.add_uint32("qwen3_asr.audio.n_window",                int(ac.get("n_window", 50)))
    writer.add_uint32("qwen3_asr.audio.n_window_infer",          int(ac.get("n_window_infer", 800)))
    writer.add_uint32("qwen3_asr.audio.conv_chunksize",          int(ac.get("conv_chunksize", 500)))
    writer.add_uint32("qwen3_asr.audio.output_dim",              int(ac.get("output_dim", 1024)))

    # Text decoder config
    writer.add_uint32("qwen3_asr.text.vocab_size",           int(xt.get("vocab_size", 151936)))
    writer.add_uint32("qwen3_asr.text.hidden_size",          int(xt.get("hidden_size", 1024)))
    writer.add_uint32("qwen3_asr.text.intermediate_size",    int(xt.get("intermediate_size", 3072)))
    writer.add_uint32("qwen3_asr.text.num_hidden_layers",    int(xt.get("num_hidden_layers", 28)))
    writer.add_uint32("qwen3_asr.text.num_attention_heads",  int(xt.get("num_attention_heads", 16)))
    writer.add_uint32("qwen3_asr.text.num_key_value_heads",  int(xt.get("num_key_value_heads", 8)))
    writer.add_uint32("qwen3_asr.text.head_dim",             int(xt.get("head_dim", 128)))
    writer.add_float64("qwen3_asr.text.rms_norm_eps",        float(xt.get("rms_norm_eps", 1e-6)))
    writer.add_float64("qwen3_asr.text.rope_theta",          float(xt.get("rope_theta", 1_000_000.0)))
    writer.add_bool("qwen3_asr.text.tie_word_embeddings",    bool(xt.get("tie_word_embeddings", True)))
    writer.add_array("qwen3_asr.text.mrope_section",         [int(x) for x in mrope_section])

    # Special token IDs
    writer.add_uint32("qwen3_asr.audio_start_token_id", int(tc.get("audio_start_token_id", 151669)))
    writer.add_uint32("qwen3_asr.audio_end_token_id",   int(tc.get("audio_end_token_id", 151670)))
    writer.add_uint32("qwen3_asr.audio_token_id",       int(tc.get("audio_token_id", 151676)))

    # Embedded tokenizer
    writer.add_string("tokenizer.huggingface.json", tokenizer_json_str)

    # ── Tensors ─────────────────────────────────────────────────────────────────
    print("Quantizing and writing tensors ...", flush=True)
    n_tensors = len(tensors)
    for idx, (name, data) in enumerate(sorted(tensors.items()), 1):
        storage = classify_tensor(name, data.shape)

        if storage == "q8_0":
            # Convert BF16/F32 → float32, then quantize to Q8_0.
            # raw_shape = ORIGINAL tensor shape (NOT reversed): write_ti_data_to_file
            # already writes it in reversed order (shape[n-1-j]), and candle then
            # reverses again on read.  Passing it pre-reversed causes double reversal.
            arr = data.astype(np.float32)
            q = to_q8_0(arr)
            writer.add_tensor(
                name, q,
                raw_dtype=GGMLQuantizationType.Q8_0,
                raw_shape=list(data.shape),
            )
        elif storage == "bf16":
            # numpy has no native BF16; represent as uint16 by truncating
            # the lower 16 bits of each float32 value (standard BF16 encoding).
            arr_f32 = data.astype(np.float32)
            arr_u16 = (arr_f32.view(np.uint32) >> 16).astype(np.uint16)
            writer.add_tensor(name, arr_u16, raw_dtype=GGMLQuantizationType.BF16)
        elif storage == "f16":
            arr = data.astype(np.float16)
            # No raw_shape needed: arr.shape == data.shape, and the library
            # automatically stores list(reversed(arr.shape)) as the GGUF shape.
            writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F16)
        else:  # f32
            arr = data.astype(np.float32)
            writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)

        if idx % 50 == 0 or idx == n_tensors:
            print(f"  [{idx}/{n_tensors}] {name} {data.shape} -> {storage}", flush=True)

    print("Writing header ...", flush=True)
    writer.write_header_to_file()
    print("Writing KV data ...", flush=True)
    writer.write_kv_data_to_file()
    print("Writing tensors ...", flush=True)
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\nDone! {output_path}  ({size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-ASR safetensors to GGUF Q8_0")
    parser.add_argument("model_dir", type=Path, help="Directory with safetensors + config.json")
    parser.add_argument("output", type=Path, help="Output .gguf file path")
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        print(f"ERROR: {args.model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    write_gguf(args.model_dir, args.output)


if __name__ == "__main__":
    main()
