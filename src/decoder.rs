use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax;
use std::collections::HashMap;

use crate::config::TextDecoderConfig;

fn get_weight(weights: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    weights
        .get(name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("weight not found: {}", name))
}

// ─── RMS Norm ────────────────────────────────────────────────────────────────

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, &format!("{}.weight", prefix))?,
            eps,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let last_dim = x_f32.rank() - 1;
        let norm_sq = x_f32.sqr()?.mean_keepdim(last_dim)?;
        let inv_rms = (norm_sq + self.eps)?.sqrt()?.recip()?;
        let x_normed = x_f32.broadcast_mul(&inv_rms)?;
        x_normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
            .map_err(Into::into)
    }
}

// ─── Linear ──────────────────────────────────────────────────────────────────

struct Linear {
    weight: Tensor,
}

impl Linear {
    fn from_weight(weight: Tensor) -> Self {
        Self { weight }
    }

    fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: get_weight(weights, &format!("{}.weight", prefix))?,
        })
    }

    // Use the weight's native dtype (BF16) for matmul to match the reference.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = self.weight.dtype();
        let x_cast = x.to_dtype(dtype)?;
        let out_features = self.weight.dims()[0];
        let dims = x_cast.dims().to_vec();
        let in_features = dims[dims.len() - 1];
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let x_2d = x_cast.reshape((batch, in_features))?;
        let out_2d = x_2d.matmul(&self.weight.t()?)?;
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(out_features);
        Ok(out_2d.reshape(out_shape)?)
    }
}

// ─── MRoPE ───────────────────────────────────────────────────────────────────

pub fn compute_mrope_cos_sin(
    position_ids: &[Vec<i64>; 3],
    head_dim: usize,
    rope_theta: f64,
    mrope_section: &[usize],
    interleaved: bool,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let seq_len = position_ids[0].len();

    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    let dim_map = if interleaved {
        build_interleaved_dim_map(mrope_section, half_dim)
    } else {
        build_contiguous_dim_map(mrope_section, half_dim)
    };

    let mut cos_vals = vec![0.0f32; seq_len * head_dim];
    let mut sin_vals = vec![0.0f32; seq_len * head_dim];

    for t in 0..seq_len {
        for j in 0..half_dim {
            let dim = dim_map[j];
            let pos = position_ids[dim][t] as f64;
            let angle = pos * inv_freq[j];
            let c = angle.cos() as f32;
            let s = angle.sin() as f32;
            cos_vals[t * head_dim + j] = c;
            sin_vals[t * head_dim + j] = s;
            cos_vals[t * head_dim + j + half_dim] = c;
            sin_vals[t * head_dim + j + half_dim] = s;
        }
    }

    let cos = Tensor::from_vec(cos_vals, (seq_len, head_dim), device)?;
    let sin = Tensor::from_vec(sin_vals, (seq_len, head_dim), device)?;

    Ok((cos, sin))
}

fn build_contiguous_dim_map(sections: &[usize], total: usize) -> Vec<usize> {
    let mut map = Vec::with_capacity(total);
    for (dim, &size) in sections.iter().enumerate() {
        for _ in 0..size {
            if map.len() >= total {
                break;
            }
            map.push(dim);
        }
    }
    while map.len() < total {
        map.push(sections.len() - 1);
    }
    map
}

fn build_interleaved_dim_map(sections: &[usize], total: usize) -> Vec<usize> {
    let n_dims = sections.len();
    let mut map = Vec::with_capacity(total);
    let mut counts = vec![0usize; n_dims];

    while map.len() < total {
        let prev_len = map.len();
        for dim in 0..n_dims {
            if map.len() >= total {
                break;
            }
            if counts[dim] < sections[dim] {
                map.push(dim);
                counts[dim] += 1;
            }
        }
        if map.len() == prev_len {
            break;
        }
    }
    map
}

/// Apply rotary embeddings.
/// `cos` and `sin` must already be cast to `x.dtype()` and shaped [1, 1, seq, head_dim].
/// The cast and unsqueeze are done once upstream in `TextDecoder::forward`.
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let x_rotated = rotate_half(x)?;
    (x.broadcast_mul(cos)? + x_rotated.broadcast_mul(sin)?).map_err(Into::into)
}

fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dims()[x.rank() - 1];
    let half = last_dim / 2;
    let x1 = x.narrow(x.rank() - 1, 0, half)?;
    let x2 = x.narrow(x.rank() - 1, half, half)?;
    let neg_x2 = (x2 * (-1.0f64))?;
    Tensor::cat(&[&neg_x2, &x1], x.rank() - 1).map_err(Into::into)
}

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x); // consume ownership: no clone needed
    }
    let (bsz, num_kv, seq, hd) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((bsz, num_kv, n_rep, seq, hd))?
        .reshape((bsz, num_kv * n_rep, seq, hd))
        .map_err(Into::into)
}

// ─── KV Cache ─────────────────────────────────────────────────────────────────

pub struct KvCache {
    pub layers: Vec<Option<(Tensor, Tensor)>>,
}

impl KvCache {
    pub fn new(num_layers: usize) -> Self {
        Self { layers: vec![None; num_layers] }
    }

    pub fn get(&self, layer: usize) -> Option<&(Tensor, Tensor)> {
        self.layers[layer].as_ref()
    }

    pub fn set(&mut self, layer: usize, cache: (Tensor, Tensor)) {
        self.layers[layer] = Some(cache);
    }

    pub fn seq_len(&self) -> usize {
        self.layers[0]
            .as_ref()
            .map(|(k, _)| k.dims()[2])
            .unwrap_or(0)
    }
}

// ─── Text Attention (GQA + QK-norm + MRoPE) ───────────────────────────────────

struct TextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl TextAttention {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: Linear::load(weights, &format!("{}.q_proj", prefix))?,
            k_proj: Linear::load(weights, &format!("{}.k_proj", prefix))?,
            v_proj: Linear::load(weights, &format!("{}.v_proj", prefix))?,
            o_proj: Linear::load(weights, &format!("{}.o_proj", prefix))?,
            q_norm: RmsNorm::load(weights, &format!("{}.q_norm", prefix), rms_norm_eps)?,
            k_norm: RmsNorm::load(weights, &format!("{}.k_norm", prefix), rms_norm_eps)?,
            num_q_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: Option<&(Tensor, Tensor)>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let (bsz, seq_len, _) = x.dims3()?;
        let nqh = self.num_q_heads;
        let nkvh = self.num_kv_heads;
        let hd = self.head_dim;

        let q = self.q_proj.forward(x)?.reshape((bsz, seq_len, nqh, hd))?.transpose(1, 2)?.contiguous()?;
        let k = self.k_proj.forward(x)?.reshape((bsz, seq_len, nkvh, hd))?.transpose(1, 2)?.contiguous()?;
        let v = self.v_proj.forward(x)?.reshape((bsz, seq_len, nkvh, hd))?.transpose(1, 2)?.contiguous()?;

        // QK normalization (applied per-head, on last dim)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Apply RoPE
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // Append to KV cache
        let (k, v) = if let Some((past_k, past_v)) = kv_cache {
            let k = Tensor::cat(&[past_k, &k], 2)?;
            let v = Tensor::cat(&[past_v, &v], 2)?;
            (k, v)
        } else {
            (k, v)
        };

        // Save pre-expand K/V as the cache entry, then consume originals for repeat_kv.
        // This avoids a second clone inside repeat_kv when n_rep == 1.
        let n_rep = nqh / nkvh;
        let new_cache = (k.clone(), v.clone());
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        // Attention
        let scale = (hd as f64).sqrt();
        let mut attn: Tensor = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * (1.0 / scale))?;

        if let Some(m) = mask {
            // Cast mask to attn's dtype (mask is F32, attn may be BF16).
            attn = attn.broadcast_add(&m.to_dtype(attn.dtype())?)?;
        }

        // Softmax in F32 for numerical stability, cast back to compute dtype.
        let attn_dtype = attn.dtype();
        let attn = softmax(&attn.to_dtype(DType::F32)?, 3)?.to_dtype(attn_dtype)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((bsz, seq_len, nqh * hd))?;
        let out = self.o_proj.forward(&out)?;

        Ok((out, new_cache))
    }
}

// ─── SwiGLU MLP ───────────────────────────────────────────────────────────────

struct TextMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl TextMlp {
    fn load(weights: &HashMap<String, Tensor>, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate_proj: Linear::load(weights, &format!("{}.gate_proj", prefix))?,
            up_proj: Linear::load(weights, &format!("{}.up_proj", prefix))?,
            down_proj: Linear::load(weights, &format!("{}.down_proj", prefix))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ─── Text Decoder Layer ───────────────────────────────────────────────────────

struct TextDecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: TextAttention,
    post_attention_layernorm: RmsNorm,
    mlp: TextMlp,
}

impl TextDecoderLayer {
    fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::load(
                weights,
                &format!("{}.input_layernorm", prefix),
                rms_norm_eps,
            )?,
            self_attn: TextAttention::load(
                weights,
                &format!("{}.self_attn", prefix),
                num_q_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
            )?,
            post_attention_layernorm: RmsNorm::load(
                weights,
                &format!("{}.post_attention_layernorm", prefix),
                rms_norm_eps,
            )?,
            mlp: TextMlp::load(weights, &format!("{}.mlp", prefix))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: Option<&(Tensor, Tensor)>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        // Pre-norm + attention + residual
        let h = self.input_layernorm.forward(x)?;
        let (h, new_cache) = self.self_attn.forward(&h, cos, sin, kv_cache, mask)?;
        let x = (x + &h)?;

        // Pre-norm + MLP + residual
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let out = (&x + &h)?;

        Ok((out, new_cache))
    }
}

// ─── Causal mask ─────────────────────────────────────────────────────────────

pub fn create_causal_mask(seq_len: usize, past_len: usize, device: &Device) -> Result<Tensor> {
    let total_len = past_len + seq_len;
    // Create upper triangular mask: positions j > past_len + i get -inf
    let mut mask_data = vec![0.0f32; seq_len * total_len];
    for i in 0..seq_len {
        for j in 0..total_len {
            if j > past_len + i {
                mask_data[i * total_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask_data, (1, 1, seq_len, total_len), device).map_err(Into::into)
}

// ─── Text Decoder ────────────────────────────────────────────────────────────

pub struct TextDecoder {
    embed_tokens: Tensor,
    layers: Vec<TextDecoderLayer>,
    norm: RmsNorm,
    lm_head_weight: Tensor,
    config: TextDecoderConfig,
}

impl TextDecoder {
    pub fn load(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        config: &TextDecoderConfig,
    ) -> Result<Self> {
        let embed_tokens =
            get_weight(weights, &format!("{}.embed_tokens.weight", prefix))?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = TextDecoderLayer::load(
                weights,
                &format!("{}.layers.{}", prefix, i),
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rms_norm_eps,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(weights, &format!("{}.norm", prefix), config.rms_norm_eps)?;

        // LM head: tied to embed_tokens if tie_word_embeddings = true
        let lm_head_weight = if config.tie_word_embeddings {
            embed_tokens.clone()
        } else {
            let lm_head_prefix = prefix.replace(".model", ".lm_head");
            get_weight(weights, &format!("{}.weight", lm_head_prefix))?
        };

        Ok(Self { embed_tokens, layers, norm, lm_head_weight, config: config.clone() })
    }

    /// Look up token embeddings. Returns the native dtype of the embedding table (BF16).
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens
            .index_select(input_ids, 0)
            .map_err(Into::into)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor, // [1, seq, hidden]
        cos: &Tensor,           // [seq, head_dim], F32
        sin: &Tensor,           // [seq, head_dim], F32
        kv_cache: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden = hidden_states.clone();

        // Cast cos/sin to compute dtype and unsqueeze to [1, 1, seq, head_dim] once,
        // rather than repeating the cast inside apply_rotary_emb for every layer.
        let compute_dtype = hidden.dtype();
        let cos4d = cos.to_dtype(compute_dtype)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin4d = sin.to_dtype(compute_dtype)?.unsqueeze(0)?.unsqueeze(0)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = kv_cache.get(i);
            let (h, new_cache) = layer.forward(&hidden, &cos4d, &sin4d, cache, mask)?;
            kv_cache.set(i, new_cache);
            hidden = h;
        }

        let hidden = self.norm.forward(&hidden)?;
        let lm_w = &self.lm_head_weight;
        // Reshape to 2D for matmul (candle requires same-rank tensors).
        // Cast hidden to weight dtype (BF16) for matmul.
        let dims = hidden.dims().to_vec();
        let hidden_size = dims[dims.len() - 1];
        let batch: usize = dims[..dims.len() - 1].iter().product();
        let hidden_2d = hidden.to_dtype(lm_w.dtype())?.reshape((batch, hidden_size))?;
        let logits_2d = hidden_2d.matmul(&lm_w.t()?)?;
        let vocab_size = lm_w.dims()[0];
        let mut out_shape = dims[..dims.len() - 1].to_vec();
        out_shape.push(vocab_size);
        logits_2d.reshape(out_shape).map_err(Into::into)
    }

    pub fn config(&self) -> &TextDecoderConfig {
        &self.config
    }
}
