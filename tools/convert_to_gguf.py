#!/usr/bin/env python3
"""
Convert Qwen3-ASR safetensors weights to GGUF format.

Supported quantizations (--quant):
  q8_0  (default) — 8-bit symmetric, block=32, ~1.0 bpw overhead
  q4_0            — 4-bit symmetric, block=32, signed nibbles
  q4_k            — 4-bit K-quant super-block=256, sub-block scales/mins

Usage:
    pip install gguf safetensors numpy
    python tools/convert_to_gguf.py <model_dir> <output.gguf> [--quant q8_0|q4_0|q4_k]

Examples:
    python tools/convert_to_gguf.py models/         qwen3_asr_0.6b_q8_0.gguf
    python tools/convert_to_gguf.py models/         qwen3_asr_0.6b_q4_k.gguf  --quant q4_k
    python tools/convert_to_gguf.py models_1.7b/    qwen3_asr_1.7b_q8_0.gguf
    python tools/convert_to_gguf.py models_1.7b/    qwen3_asr_1.7b_q4_k.gguf  --quant q4_k
"""

import argparse
import json
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

# ─── Q8_0 ─────────────────────────────────────────────────────────────────────
# Block of 32: fp16 scale (2 bytes) + 32 × int8 quants (32 bytes) = 34 bytes

DTYPE_Q8_0 = np.dtype([("d", np.float16), ("qs", np.int8, (32,))])


def to_q8_0(data: np.ndarray) -> np.ndarray:
    """Quantize float array to Q8_0 block format (8-bit symmetric)."""
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


# ─── Q4_0 ─────────────────────────────────────────────────────────────────────
# Block of 32: fp16 scale (2 bytes) + 16 × uint8 packed nibbles (16 bytes) = 18 bytes
#
# Encoding (matches ggml quantize_row_q4_0_ref + candle k_quants.rs):
#   d  = signed_max / -8    (signed_max = element with largest |x|, sign preserved)
#   id = 1 / d
#   q_i = clamp(floor(x_i * id + 8.5), 0, 15)   → stored as 4-bit nibble
#   Split-half packing (NOT sequential pairs!):
#     qs[j] = (nibble for element j) | ((nibble for element j+16) << 4)
# Dequant: ys[j] = d*(qs[j] & 0xF - 8), ys[j+16] = d*(qs[j]>>4 - 8)

DTYPE_Q4_0 = np.dtype([("d", np.float16), ("qs", np.uint8, (16,))])


def to_q4_0(data: np.ndarray) -> np.ndarray:
    """Quantize float array to Q4_0 block format (4-bit symmetric signed)."""
    flat = data.flatten().astype(np.float32)
    n_blocks = (len(flat) + 31) // 32
    padded = np.zeros(n_blocks * 32, dtype=np.float32)
    padded[: len(flat)] = flat
    blocks = padded.reshape(n_blocks, 32)  # [n_blocks, 32]

    # Find element with largest absolute value (preserving sign) per block.
    max_idx = np.argmax(np.abs(blocks), axis=1)          # [n_blocks]
    signed_max = blocks[np.arange(n_blocks), max_idx]    # [n_blocks]

    # d = signed_max / -8  (fp16)
    d = (signed_max / -8.0).astype(np.float16)           # [n_blocks]
    d_f32 = d.astype(np.float32)
    id_f32 = np.where(d_f32 != 0.0, 1.0 / d_f32, 0.0)  # [n_blocks]

    # Quantize to nibbles [0, 15]
    q = np.clip(
        np.floor(blocks * id_f32[:, None] + 8.5).astype(np.int32),
        0, 15,
    ).astype(np.uint8)  # [n_blocks, 32]

    # Candle/ggml split-half packing:
    #   qs[j] low nibble  = element j        (first  half, indices  0..15)
    #   qs[j] high nibble = element j + 16   (second half, indices 16..31)
    qs = (q[:, :16] & 0xF) | ((q[:, 16:].astype(np.uint16) & 0xF) << 4).astype(np.uint8).astype(np.uint8)

    result = np.empty(n_blocks, dtype=DTYPE_Q4_0)
    result["d"] = d
    result["qs"] = qs
    return result


# ─── Q4_K ─────────────────────────────────────────────────────────────────────
# Super-block of 256: fp16 d (2) + fp16 dmin (2) + uint8[12] scales + uint8[128] qs = 144 bytes
#
# Layout:
#   8 sub-blocks of 32 elements each.
#   For each sub-block k: sub-block scale sc[k] and sub-block min m[k] (both ≥ 0).
#   d     = max(sc) / 63   (super-block scale for sub-block scales, fp16)
#   dmin  = max(m)  / 63   (super-block scale for sub-block mins,   fp16)
#   sc_q[k] = round(sc[k] / d)    ∈ [0, 63]  (6-bit)
#   m_q[k]  = round(m[k]  / dmin) ∈ [0, 63]  (6-bit)
#
# Scales packing (12 bytes, matches ggml get_scale_min_k4 layout):
#   For j in 0..3:
#     scales[j]   = (sc_q[j] & 0x3F) | ((sc_q[j+4] & 0x30) << 2)
#     scales[j+4] = (m_q[j]  & 0x3F) | ((m_q[j+4]  & 0x30) << 2)
#     scales[j+8] = (sc_q[j+4] & 0x0F) | ((m_q[j+4] & 0x0F) << 4)
#
# 4-bit quant per element i in sub-block k:
#   q_i = clamp(round((x_i + m_q[k]*dmin) / (sc_q[k]*d) * 15), 0, 15)
#
# Nibble packing (sub-block pairs {0,1}, {2,3}, {4,5}, {6,7}):
#   qs[pair*32 : pair*32+32] = L[pair*64 : pair*64+32] | (L[pair*64+32 : pair*64+64] << 4)
# Dequant: x = d * sc_q[k] * nibble - dmin * m_q[k]

QK_K = 256
DTYPE_Q4_K = np.dtype([
    ("d",      np.float16),
    ("dmin",   np.float16),
    ("scales", np.uint8, (12,)),
    ("qs",     np.uint8, (128,)),
])


def to_q4_k(data: np.ndarray) -> np.ndarray:
    """Quantize float array to Q4_K super-block format (4-bit K-quant)."""
    flat = data.flatten().astype(np.float32)
    n_super = (len(flat) + QK_K - 1) // QK_K
    padded = np.zeros(n_super * QK_K, dtype=np.float32)
    padded[: len(flat)] = flat

    # [n_super, 8, 32] — 8 sub-blocks of 32 elements
    blocks = padded.reshape(n_super, 8, 32)

    # Per-sub-block step size (scale = range/15) and flipped min: [n_super, 8]
    # ggml dequant: x = d * sc_q * nibble - dmin * m_q
    # → sc must be the per-element step (range / 15), not the full range.
    block_min = blocks.min(axis=2)
    block_max = blocks.max(axis=2)
    sc_arr = np.maximum((block_max - block_min) / 15.0, 0.0)  # step size ≥ 0
    m_arr  = np.maximum(-block_min,                    0.0)   # abs-min ≥ 0

    # Super-block fp16 scales
    max_sc   = sc_arr.max(axis=1)           # [n_super]
    max_m    = m_arr.max(axis=1)            # [n_super]
    d_f32    = np.where(max_sc > 0, max_sc / 63.0, 0.0)
    dmin_f32 = np.where(max_m  > 0, max_m  / 63.0, 0.0)
    d_f16    = d_f32.astype(np.float16)
    dmin_f16 = dmin_f32.astype(np.float16)

    # Quantize sub-block scales/mins to 6-bit: [n_super, 8]
    safe_d    = np.where(d_f32[:, None]    > 0, d_f32[:, None],    1.0)
    safe_dmin = np.where(dmin_f32[:, None] > 0, dmin_f32[:, None], 1.0)
    sc_q = np.clip(np.round(sc_arr / safe_d),    0, 63).astype(np.uint8)
    m_q  = np.clip(np.round(m_arr  / safe_dmin), 0, 63).astype(np.uint8)

    # Pack 8 scales + 8 mins (6-bit each) into 12 bytes per super-block.
    # Use uint16 intermediates to avoid uint8 overflow on shifts.
    sc_lo = sc_q[:, :4].astype(np.uint16)   # sc[0..3]  [n_super, 4]
    sc_hi = sc_q[:, 4:].astype(np.uint16)   # sc[4..7]  [n_super, 4]
    m_lo  = m_q[:,  :4].astype(np.uint16)   # m[0..3]   [n_super, 4]
    m_hi  = m_q[:,  4:].astype(np.uint16)   # m[4..7]   [n_super, 4]

    s0_3  = ((sc_lo & 0x3F) | ((sc_hi & 0x30) << 2)).astype(np.uint8)   # [n_super, 4]
    s4_7  = ((m_lo  & 0x3F) | ((m_hi  & 0x30) << 2)).astype(np.uint8)   # [n_super, 4]
    s8_11 = ((sc_hi & 0x0F) | ((m_hi  & 0x0F) << 4)).astype(np.uint8)   # [n_super, 4]
    scales_out = np.concatenate([s0_3, s4_7, s8_11], axis=1)             # [n_super, 12]

    # Quantize values to 4-bit using dequantized per-sub-block scale and min.
    # sc_eff[n, k] = sc_q[n,k] * d[n],  m_eff[n, k] = m_q[n,k] * dmin[n]
    sc_eff = sc_q.astype(np.float32) * d_f32[:, None]      # [n_super, 8]
    m_eff  = m_q.astype(np.float32)  * dmin_f32[:, None]   # [n_super, 8]
    safe_sc = np.where(sc_eff[:, :, None] > 0, sc_eff[:, :, None], 1.0)

    # L[n, k, i] = clamp(round((x + m_eff) / sc_eff), 0, 15)
    # sc_eff ≈ step size = (max-min)/15, so (x - min) / step → [0, 15]
    L = np.clip(
        np.round((blocks + m_eff[:, :, None]) / safe_sc),
        0, 15,
    ).astype(np.uint8)  # [n_super, 8, 32]

    # Pack nibbles: sub-block pairs {0,1}, {2,3}, {4,5}, {6,7}
    # qs[pair*32 : pair*32+32] = L[:,2*pair,:] | (L[:,2*pair+1,:] << 4)
    L_pairs = L.reshape(n_super, 4, 2, 32)          # [n_super, 4, 2, 32]
    lo = L_pairs[:, :, 0, :].astype(np.uint16)      # [n_super, 4, 32]
    hi = L_pairs[:, :, 1, :].astype(np.uint16)
    qs_out = ((lo & 0xF) | ((hi & 0xF) << 4)).astype(np.uint8).reshape(n_super, 128)

    result = np.empty(n_super, dtype=DTYPE_Q4_K)
    result["d"]      = d_f16
    result["dmin"]   = dmin_f16
    result["scales"] = scales_out
    result["qs"]     = qs_out
    return result


# ─── Safetensors loading ───────────────────────────────────────────────────────

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


# ─── Tensor classification ─────────────────────────────────────────────────────

def classify_tensor(name: str, shape: tuple, quant_type: str) -> str:
    """Return storage type based on tensor shape and selected quantization.

    Policy (mirrors the original BF16 safetensors model):
      - 1D tensors (norms, biases)  → F32   (always; numerical stability)
      - 4D tensors (conv weights)   → BF16  (always; too small / spatial)
      - 2D weight matrices          → <quant_type> (q8_0, q4_0, or q4_k)

    Candle / GGUFReader compatibility note:
      Both check that shape[-1] (innermost / column dimension in C-order)
      is divisible by the quantization block size.
      Q4_K uses QK_K=256; tensors where shape[-1] % 256 != 0 fall back to
      Q8_0 (block_size=32) which divides all common audio encoder dims.
      Example: 0.6B audio encoder d_model=896 → 896 % 256 = 128 ≠ 0.
    """
    ndim = len(shape)

    if ndim == 1:
        return "f32"

    if ndim == 2:
        if quant_type == "q4_k" and shape[-1] % QK_K != 0:
            # Fall back to Q8_0 for dims not divisible by 256
            return "q8_0"
        return quant_type

    # 4D conv weights and anything else → BF16
    return "bf16"


# ─── Model name ───────────────────────────────────────────────────────────────

def get_model_name(model_dir: Path, config: dict) -> str:
    """Determine a human-readable model name."""
    dir_name = model_dir.resolve().name.lower()
    if "1.7b" in dir_name or "1_7b" in dir_name:
        return "Qwen3-ASR-1.7B"
    hidden_size = (
        config.get("thinker_config", {})
        .get("text_config", {})
        .get("hidden_size", 1024)
    )
    if hidden_size > 1024:
        return "Qwen3-ASR-1.7B"
    return "Qwen3-ASR-0.6B"


# ─── Main conversion ──────────────────────────────────────────────────────────

def write_gguf(model_dir: Path, output_path: Path, quant_type: str) -> None:
    print(f"Loading config from {model_dir} ...", flush=True)
    config = load_config(model_dir)

    tc = config.get("thinker_config", {})
    ac = tc.get("audio_config", {})
    xt = tc.get("text_config", {})

    tokenizer_path = model_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
    with open(tokenizer_path, encoding="utf-8") as f:
        tokenizer_json_str = f.read()

    model_name = get_model_name(model_dir, config)
    print(f"Model: {model_name}  quant: {quant_type}", flush=True)

    print(f"Loading safetensors from {model_dir} ...", flush=True)
    tensors = load_safetensors(model_dir)
    print(f"Loaded {len(tensors)} tensors", flush=True)

    rope_scaling = xt.get("rope_scaling", {}) or {}
    mrope_section = rope_scaling.get("mrope_section", [24, 20, 20])

    print(f"Creating GGUF writer -> {output_path}", flush=True)
    writer = GGUFWriter(str(output_path), arch="qwen3_asr")

    # ── Metadata ────────────────────────────────────────────────────────────────
    writer.add_name(model_name)

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

    writer.add_uint32("qwen3_asr.audio_start_token_id", int(tc.get("audio_start_token_id", 151669)))
    writer.add_uint32("qwen3_asr.audio_end_token_id",   int(tc.get("audio_end_token_id",   151670)))
    writer.add_uint32("qwen3_asr.audio_token_id",       int(tc.get("audio_token_id",       151676)))

    writer.add_string("tokenizer.huggingface.json", tokenizer_json_str)

    # ── Tensors ─────────────────────────────────────────────────────────────────
    print(f"Quantizing and writing tensors (quant={quant_type}) ...", flush=True)
    n_tensors = len(tensors)
    for idx, (name, data) in enumerate(sorted(tensors.items()), 1):
        storage = classify_tensor(name, data.shape, quant_type)

        if storage == "q8_0":
            arr = data.astype(np.float32)
            q = to_q8_0(arr)
            writer.add_tensor(
                name, q,
                raw_dtype=GGMLQuantizationType.Q8_0,
                raw_shape=list(data.shape),
            )

        elif storage == "q4_0":
            arr = data.astype(np.float32)
            q = to_q4_0(arr)
            writer.add_tensor(
                name, q,
                raw_dtype=GGMLQuantizationType.Q4_0,
                raw_shape=list(data.shape),
            )

        elif storage == "q4_k":
            arr = data.astype(np.float32)
            q = to_q4_k(arr)
            writer.add_tensor(
                name, q,
                raw_dtype=GGMLQuantizationType.Q4_K,
                raw_shape=list(data.shape),
            )

        elif storage == "bf16":
            arr_f32 = data.astype(np.float32)
            arr_u16 = (arr_f32.view(np.uint32) >> 16).astype(np.uint16)
            writer.add_tensor(name, arr_u16, raw_dtype=GGMLQuantizationType.BF16)

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
    parser = argparse.ArgumentParser(
        description="Convert Qwen3-ASR safetensors to GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_dir", type=Path, help="Directory with safetensors + config.json")
    parser.add_argument("output",    type=Path, help="Output .gguf file path")
    parser.add_argument(
        "--quant",
        choices=["q8_0", "q4_0", "q4_k"],
        default="q8_0",
        help="Quantization type for 2D weight matrices (default: q8_0)",
    )
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        print(f"ERROR: {args.model_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    write_gguf(args.model_dir, args.output, args.quant)


if __name__ == "__main__":
    main()
