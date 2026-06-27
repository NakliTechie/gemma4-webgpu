/**
 * Qwen3 configuration (dense: 0.6B/1.7B/4B/8B) for the gemma4-webgpu engine.
 *
 * Qwen3 is a strict simplification of Gemma 4 + a SwiGLU MLP swap:
 *   - full-causal attention everywhere (no sliding window)
 *   - single RoPE theta (no dual global/swa)
 *   - uniform head_dim (no mixed 256/512)
 *   - no KV sharing (every layer is its own producer)
 *   - no per-layer embeddings (PLE)          → per_layer_input_dim = 0
 *   - no final-logit softcap                 → final_logit_softcapping = 0
 *   - per-head QK-norm (q_norm/k_norm)       → REUSED from Gemma (same kernel)
 *   - NO V-norm                              → v_norm = false
 *   - pre-norm only (no post-block sandwich) → post_attn_norm/post_ffn_norm = false
 *   - SwiGLU/SiLU MLP                         → ffn_activation = 'silu'
 *   - no embedding scaling                    → embedding_scale = 1.0
 *
 * `configFromGGUFQwen3` reads `qwen3.*` metadata and overrides the baseline.
 */
import type { GemmaConfig, GGUFParsed } from '../types.js';
import { kvNumberOrNull } from '../gguf.js';

/** Baseline Qwen3-4B config. Overridden from GGUF metadata by configFromGGUFQwen3. */
export function defaultQwen3Config(): GemmaConfig {
  const num_layers = 36;          // Qwen3-4B (1.7B=28, 0.6B=28, 8B=36)
  const head_dim = 128;           // uniform across the Qwen3 dense family
  const num_q_heads = 32;
  const num_kv_heads = 8;
  const intermediate = 9728;      // Qwen3-4B (1.7B=6144)

  return {
    hidden_size: 2560,
    q_dim: num_q_heads * head_dim,
    kv_dim: num_kv_heads * head_dim,
    num_q_heads,
    num_kv_heads,
    head_dim,
    intermediate_size: intermediate,
    vocab_size: 151936,
    num_layers,
    context_length: 2048,
    rms_norm_eps: 1e-6,
    rope_theta_global: 1_000_000.0,
    rope_theta_swa: 1_000_000.0,            // single theta ⇒ same as global
    head_dim_local: head_dim,
    head_dim_global: head_dim,              // uniform ⇒ getHeadDim() == head_dim everywhere
    sliding_window: 0,
    attention_is_sliding: new Array<boolean>(num_layers).fill(false), // full-causal everywhere
    intermediate_sizes: new Array<number>(num_layers).fill(intermediate),
    num_unshared_layers: num_layers,        // no KV sharing ⇒ every layer is a producer
    kv_producer_for_layer: Array.from({ length: num_layers }, (_, i) => i),
    per_layer_input_dim: 0,                 // no PLE
    final_logit_softcapping: 0,             // no softcap
    // ── Qwen3 variant flags ──
    arch: 'qwen3',
    ffn_activation: 'silu',
    v_norm: false,
    post_attn_norm: false,
    post_ffn_norm: false,
    embedding_scale: 1.0,                    // Qwen3 does not scale embeddings
  };
}

/** Hydrate a Qwen3 GemmaConfig from `qwen3.*` GGUF metadata. */
export function configFromGGUFQwen3(gguf: GGUFParsed, maxContextLength?: number): GemmaConfig {
  const isQwen3 = gguf.kv.has('qwen3.block_count') || gguf.kv.has('qwen3.embedding_length');
  if (!isQwen3) throw new Error('Not a Qwen3 GGUF — no `qwen3.*` metadata keys found.');

  const config = defaultQwen3Config();

  const hidden = kvNumberOrNull(gguf, 'qwen3.embedding_length');
  if (hidden !== null) config.hidden_size = hidden;

  const layers = kvNumberOrNull(gguf, 'qwen3.block_count');
  if (layers !== null) config.num_layers = layers;

  const qHeads = kvNumberOrNull(gguf, 'qwen3.attention.head_count');
  if (qHeads !== null) config.num_q_heads = qHeads;

  const kvHeads = kvNumberOrNull(gguf, 'qwen3.attention.head_count_kv');
  if (kvHeads !== null) config.num_kv_heads = kvHeads;

  // Qwen3 carries an explicit key_length (head_dim, 128) decoupled from hidden/heads.
  const keyLen = kvNumberOrNull(gguf, 'qwen3.attention.key_length');
  if (keyLen !== null) {
    config.head_dim = keyLen;
    config.head_dim_local = keyLen;
    config.head_dim_global = keyLen;
  }

  const ffn = kvNumberOrNull(gguf, 'qwen3.feed_forward_length');
  if (ffn !== null) {
    config.intermediate_size = ffn;
  }

  const eps = kvNumberOrNull(gguf, 'qwen3.attention.layer_norm_rms_epsilon');
  if (eps !== null) config.rms_norm_eps = eps;

  const theta = kvNumberOrNull(gguf, 'qwen3.rope.freq_base');
  if (theta !== null) { config.rope_theta_global = theta; config.rope_theta_swa = theta; }

  const maxCtx = maxContextLength ?? 2048;
  const ctxLen = kvNumberOrNull(gguf, 'qwen3.context_length');
  if (ctxLen !== null) config.context_length = Math.min(ctxLen, maxCtx);

  const vocab = kvNumberOrNull(gguf, 'qwen3.vocab_size');
  if (vocab !== null) config.vocab_size = vocab;

  // Untied LM head: present in larger Qwen3 GGUFs as `output.weight`. If absent,
  // the model ties embeddings (token_embd doubles as the LM head).
  const hasOutput = gguf.tensors.some((t) => t.name === 'output.weight');
  config.lm_head_tensor = hasOutput ? 'output' : undefined;

  // Recompute derived aggregates + per-layer arrays for the (possibly new) layer count.
  config.q_dim = config.num_q_heads * config.head_dim;
  config.kv_dim = config.num_kv_heads * config.head_dim;
  config.attention_is_sliding = new Array<boolean>(config.num_layers).fill(false);
  config.intermediate_sizes = new Array<number>(config.num_layers).fill(config.intermediate_size);
  config.num_unshared_layers = config.num_layers;
  config.kv_producer_for_layer = Array.from({ length: config.num_layers }, (_, i) => i);

  return config;
}
