use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Module};
use candle_nn::ops::softmax_last_dim;
use std::collections::HashMap;

use crate::config::AudioEncoderConfig;
use crate::linear::LinearW;

// ─── Weight helpers (dense / safetensors path) ────────────────────────────────

fn get_w(weights: &HashMap<String, Tensor>, name: &str) -> anyhow::Result<Tensor> {
    weights
        .get(name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))
}

fn load_linear(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<LinearW> {
    Ok(LinearW::new(
        get_w(weights, &format!("{}.weight", prefix))?,
        weights.get(&format!("{}.bias", prefix)).cloned(),
    ))
}

fn load_layer_norm(weights: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<LayerNorm> {
    Ok(LayerNorm::new(
        get_w(weights, &format!("{}.weight", prefix))?,
        get_w(weights, &format!("{}.bias", prefix))?,
        eps,
    ))
}

fn load_conv2d(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    stride: usize,
    padding: usize,
) -> Result<Conv2d> {
    Ok(Conv2d::new(
        get_w(weights, &format!("{}.weight", prefix))?,
        weights.get(&format!("{}.bias", prefix)).cloned(),
        Conv2dConfig { stride, padding, ..Default::default() },
    ))
}

// ─── Audio Encoder Self-Attention ─────────────────────────────────────────────

struct AudioAttention {
    q_proj: LinearW,
    k_proj: LinearW,
    v_proj: LinearW,
    out_proj: LinearW,
    num_heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_heads: usize,
        d_model: usize,
    ) -> Result<Self> {
        let head_dim = d_model / num_heads;
        Ok(Self {
            q_proj:   load_linear(weights, &format!("{}.q_proj", prefix))?,
            k_proj:   load_linear(weights, &format!("{}.k_proj", prefix))?,
            v_proj:   load_linear(weights, &format!("{}.v_proj", prefix))?,
            out_proj: load_linear(weights, &format!("{}.out_proj", prefix))?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;
        let nh = self.num_heads;
        let hd = self.head_dim;

        let q = self.q_proj.forward(x)?
            .reshape((bsz, seq_len, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self.k_proj.forward(x)?
            .reshape((bsz, seq_len, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self.v_proj.forward(x)?
            .reshape((bsz, seq_len, nh, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = (hd as f64).sqrt();
        let mut attn: Tensor = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * (1.0 / scale))?;

        if let Some(m) = mask {
            attn = attn.broadcast_add(m)?;
        }

        let attn = softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((bsz, seq_len, nh * hd))?;
        self.out_proj.forward(&out).map_err(Into::into)
    }
}

// ─── Audio Encoder FFN ────────────────────────────────────────────────────────

struct AudioFfn {
    fc1: LinearW,
    fc2: LinearW,
}

impl AudioFfn {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: load_linear(weights, &format!("{}.fc1", prefix))?,
            fc2: load_linear(weights, &format!("{}.fc2", prefix))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?).map_err(Into::into)
    }
}

// ─── Audio Encoder Layer ──────────────────────────────────────────────────────

struct AudioEncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: AudioAttention,
    final_layer_norm: LayerNorm,
    ffn: AudioFfn,
}

impl AudioEncoderLayer {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_heads: usize,
        d_model: usize,
    ) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: load_layer_norm(
                weights,
                &format!("{}.self_attn_layer_norm", prefix),
                1e-5,
            )?,
            self_attn: AudioAttention::load(
                weights,
                &format!("{}.self_attn", prefix),
                num_heads,
                d_model,
            )?,
            final_layer_norm: load_layer_norm(
                weights,
                &format!("{}.final_layer_norm", prefix),
                1e-5,
            )?,
            ffn: AudioFfn::load(weights, prefix)?,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm + self-attention + residual
        let h = self.self_attn_layer_norm.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;
        let x = (x + &h)?;

        // Pre-norm + FFN + residual
        let h = self.final_layer_norm.forward(&x)?;
        let h = self.ffn.forward(&h)?;
        (&x + &h).map_err(Into::into)
    }
}

// ─── Sinusoidal positional embedding ─────────────────────────────────────────

fn create_sinusoidal_embedding(max_len: usize, dim: usize, device: &Device) -> Result<Tensor> {
    let half_dim = dim / 2;
    let log_timescale = (10000.0f64).ln() / (half_dim as f64 - 1.0);

    let mut embeddings = vec![0.0f32; max_len * dim];
    for pos in 0..max_len {
        for i in 0..half_dim {
            let inv_ts = (-(i as f64) * log_timescale).exp();
            let angle = pos as f64 * inv_ts;
            embeddings[pos * dim + i] = angle.sin() as f32;
            embeddings[pos * dim + half_dim + i] = angle.cos() as f32;
        }
    }

    Tensor::from_vec(embeddings, (max_len, dim), device).map_err(Into::into)
}

// ─── Audio Encoder ────────────────────────────────────────────────────────────

pub(crate) struct AudioEncoder {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: LinearW,
    positional_embedding: Tensor,
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,
    proj1: LinearW,
    proj2: LinearW,
    config: AudioEncoderConfig,
}

impl AudioEncoder {
    pub(crate) fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        config: &AudioEncoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let conv2d1 = load_conv2d(weights, &format!("{}.conv2d1", prefix), 2, 1)?;
        let conv2d2 = load_conv2d(weights, &format!("{}.conv2d2", prefix), 2, 1)?;
        let conv2d3 = load_conv2d(weights, &format!("{}.conv2d3", prefix), 2, 1)?;
        let conv_out = load_linear(weights, &format!("{}.conv_out", prefix))?;

        let mut layers = Vec::new();
        for i in 0..config.encoder_layers {
            let layer = AudioEncoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config.encoder_attention_heads,
                config.d_model,
            )?;
            layers.push(layer);
        }

        let ln_post = load_layer_norm(weights, &format!("{}.ln_post", prefix), 1e-5)?;
        let proj1 = load_linear(weights, &format!("{}.proj1", prefix))?;
        let proj2 = load_linear(weights, &format!("{}.proj2", prefix))?;

        let positional_embedding =
            create_sinusoidal_embedding(config.max_source_positions, config.d_model, device)?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            positional_embedding,
            layers,
            ln_post,
            proj1,
            proj2,
            config: config.clone(),
        })
    }

    /// Encode mel spectrogram [num_mel_bins, num_frames] → [num_tokens, output_dim]
    pub(crate) fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let num_frames = mel.dims()[1];
        // Logical chunk = n_window * 2 = 100 mel frames (matches official windowed attention).
        let chunk_size = self.config.n_window * 2;

        let num_full = num_frames / chunk_size;
        let tail = num_frames % chunk_size;
        let num_chunks = num_full + if tail > 0 { 1 } else { 0 };

        // Collect chunks as F32 (conv2d runs in F32).
        let mut chunk_mels: Vec<Tensor> = Vec::with_capacity(num_chunks);
        let mut chunk_valid_tokens: Vec<usize> = Vec::with_capacity(num_chunks);

        for i in 0..num_full {
            let start = i * chunk_size;
            let chunk = mel.narrow(1, start, chunk_size)?.unsqueeze(0)?;
            chunk_mels.push(chunk);
            chunk_valid_tokens.push(Self::feat_extract_output_length(chunk_size));
        }

        if tail > 0 {
            let start = num_full * chunk_size;
            let tail_mel = mel.narrow(1, start, tail)?;
            let pad_frames = chunk_size - tail;
            let device = mel.device();
            let pad = Tensor::zeros((mel.dims()[0], pad_frames), DType::F32, device)?;
            let padded = Tensor::cat(&[&tail_mel.to_dtype(DType::F32)?, &pad], 1)?.unsqueeze(0)?;
            chunk_mels.push(padded);
            chunk_valid_tokens.push(Self::feat_extract_output_length(tail));
        }

        // Stack chunks and cast to conv weight's native dtype.
        // batched: [num_chunks, 1, mel_bins, chunk_size]
        let refs: Vec<&Tensor> = chunk_mels.iter().collect();
        let compute_dtype = self.conv2d1.weight().dtype();
        let batched = Tensor::cat(&refs, 0)?.unsqueeze(1)?.to_dtype(compute_dtype)?;

        // Conv stem with GELU activations.
        let x = self.conv2d1.forward(&batched)?.gelu_erf()?;
        let x = self.conv2d2.forward(&x)?.gelu_erf()?;
        let x = self.conv2d3.forward(&x)?.gelu_erf()?;

        // Reshape: [b, c, f, t] -> [b, t, c*f]
        let (b, c, f, t) = x.dims4()?;
        let reshaped = x.permute((0, 3, 1, 2))?.contiguous()?.reshape((b, t, c * f))?;

        // Linear projection.
        let conv_out = self.conv_out.forward(&reshaped)?;

        // Add positional embedding, cast to match conv_out's dtype.
        let pos_emb = self.positional_embedding
            .narrow(0, 0, t)?
            .unsqueeze(0)?
            .to_dtype(conv_out.dtype())?;
        let conv_out = conv_out.broadcast_add(&pos_emb)?;
        // conv_out: [num_chunks, tokens_per_chunk, d_model]

        // Collect valid tokens from all chunks and concatenate.
        let mut all_valid: Vec<Tensor> = Vec::with_capacity(num_chunks);
        for (idx, &valid) in chunk_valid_tokens.iter().enumerate() {
            let chunk_tokens = conv_out.i(idx)?.narrow(0, 0, valid)?;
            all_valid.push(chunk_tokens);
        }
        // Concatenate: [total_tokens, d_model]
        let refs: Vec<&Tensor> = all_valid.iter().collect();
        let hidden = Tensor::cat(&refs, 0)?;

        // Add batch dim: [1, total_tokens, d_model]
        let mut hidden = hidden.unsqueeze(0)?;

        // Transformer encoder with full attention.
        for layer in &self.layers {
            hidden = layer.forward(&hidden, None)?;
        }

        // Output projection: LN → Linear → GELU → Linear
        let hidden = self.ln_post.forward(&hidden)?;
        let hidden = self.proj2.forward(&self.proj1.forward(&hidden)?.gelu_erf()?)?;

        // Remove batch dim: [num_tokens, output_dim]
        hidden.squeeze(0).map_err(Into::into)
    }

    fn feat_extract_output_length(input_frames: usize) -> usize {
        let after_conv = |len: usize| -> usize { (len - 1) / 2 + 1 };
        after_conv(after_conv(after_conv(input_frames)))
    }
}
