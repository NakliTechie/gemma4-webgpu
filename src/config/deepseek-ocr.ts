/**
 * Unlimited-OCR / DeepSeek-OCR decoder configuration (GGUF arch `deepseek2-ocr`).
 *
 * The decoder is a 12-layer DeepSeek-V2 MoE with — despite the family name —
 * NO MLA (`use_mla: false` in the HF config): attention is plain Llama MHA
 * (10 q heads = 10 kv heads × head_dim 128, no QK-norm, no biases, standard
 * RoPE θ=10000). Relative to the engine's Qwen3 path the deltas are:
 *   - no per-head QK-norm                     → qk_norm = false (plain rope kernel)
 *   - MoE FFN on layers ≥ 1                   → moe { 64 routed, top-6, I=896 }
 *     · layer 0 keeps a dense FFN (I=6848)      (`first_k_dense_replace: 1`)
 *     · 2 shared experts ship PRE-FUSED as one  `ffn_*_shexp` FFN (I=1792) —
 *       uploaded under the dense `ffn_gate/up/down` keys so the standard FFN
 *       path computes shared_experts(x); the routed experts add on top.
 *   - R-SWA (ring-buffer KV, window 128) is NOT implemented yet: for the
 *     first 128 generated tokens R-SWA is exactly full-causal attention, so
 *     the P0 smoke runs plain causal. Ring KV is the P2 long-doc item.
 *
 * `configFromGGUFDeepseekOcr` reads `deepseek2-ocr.*` metadata (the llama.cpp
 * PR #17400 conversion naming) and overrides the baseline.
 *
 * NOTE `rope.dimension_count = 0` in the GGUF mirrors the HF config's
 * `qk_rope_head_dim: 0`, which only applies to the (disabled) MLA path. The
 * active `SlidingWindowLlamaAttention` applies LlamaRotaryEmbedding over the
 * FULL head_dim — so we ignore that key and rotate all 128 dims.
 */
import type { GemmaConfig, GGUFParsed } from '../types.js';
import { kvNumberOrNull } from '../gguf.js';

/** Baseline Unlimited-OCR decoder config. Overridden from GGUF metadata. */
export function defaultDeepseekOcrConfig(): GemmaConfig {
  const num_layers = 12;
  const head_dim = 128;            // hidden / heads = 1280 / 10 (= HF v_head_dim)
  const num_q_heads = 10;
  const num_kv_heads = 10;         // MHA — no GQA
  const dense_intermediate = 6848; // layer 0 only
  const shared_intermediate = 1792;// 2 shared experts × 896, pre-fused in GGUF

  // Per-layer FFN width drives the REUSED dense path: layer 0 = the real
  // dense FFN, layers 1+ = the shared-expert FFN (aliased tensors).
  const intermediate_sizes = Array.from({ length: num_layers }, (_, i) =>
    i === 0 ? dense_intermediate : shared_intermediate,
  );

  return {
    hidden_size: 1280,
    q_dim: num_q_heads * head_dim,
    kv_dim: num_kv_heads * head_dim,
    num_q_heads,
    num_kv_heads,
    head_dim,
    intermediate_size: dense_intermediate,  // scalar max (work-buffer sizing)
    vocab_size: 129280,
    num_layers,
    context_length: 2048,
    rms_norm_eps: 1e-6,
    rope_theta_global: 10000.0,             // LlamaRotaryEmbedding default
    rope_theta_swa: 10000.0,
    head_dim_local: head_dim,
    head_dim_global: head_dim,
    sliding_window: 0,                      // Gemma-style mask SWA unused (R-SWA ≠ mask SWA)
    ring_window: 128,                       // R-SWA ring (HF sliding_window_size; no GGUF key)
    attention_is_sliding: new Array<boolean>(num_layers).fill(false),
    intermediate_sizes,
    num_unshared_layers: num_layers,
    kv_producer_for_layer: Array.from({ length: num_layers }, (_, i) => i),
    per_layer_input_dim: 0,                 // no PLE
    final_logit_softcapping: 0,
    // ── variant flags ──
    arch: 'deepseek-ocr',
    ffn_activation: 'silu',
    v_norm: false,
    post_attn_norm: false,
    post_ffn_norm: false,
    embedding_scale: 1.0,
    qk_norm: false,                         // plain RoPE, no per-head norm
    lm_head_tensor: 'output',               // untied lm_head (Q6_K in the GGUF)
    moe: {
      num_experts: 64,
      experts_per_tok: 6,
      moe_intermediate_size: 896,
      shared_intermediate_size: shared_intermediate,
      first_dense_layers: 1,
      norm_topk_prob: false,                // configuration_deepseek_v2 default
      routed_scaling_factor: 1.0,           // configuration_deepseek_v2 default
    },
  };
}

/** Hydrate from `deepseek2-ocr.*` GGUF metadata (llama.cpp PR #17400 naming). */
export function configFromGGUFDeepseekOcr(gguf: GGUFParsed, maxContextLength?: number): GemmaConfig {
  const P = 'deepseek2-ocr';
  const isDsOcr = gguf.kv.has(`${P}.block_count`) || gguf.kv.has(`${P}.embedding_length`);
  if (!isDsOcr) throw new Error(`Not a deepseek2-ocr GGUF — no \`${P}.*\` metadata keys found.`);

  const config = defaultDeepseekOcrConfig();
  const num = (key: string) => kvNumberOrNull(gguf, `${P}.${key}`);

  const hidden = num('embedding_length');
  if (hidden !== null) config.hidden_size = hidden;

  const layers = num('block_count');
  if (layers !== null) config.num_layers = layers;

  const qHeads = num('attention.head_count');
  if (qHeads !== null) config.num_q_heads = qHeads;

  const kvHeads = num('attention.head_count_kv');
  if (kvHeads !== null) config.num_kv_heads = kvHeads;

  // No explicit key_length key in this GGUF; head_dim = hidden / heads.
  config.head_dim = config.hidden_size / config.num_q_heads;
  config.head_dim_local = config.head_dim;
  config.head_dim_global = config.head_dim;

  const denseFfn = num('feed_forward_length');
  if (denseFfn !== null) config.intermediate_size = denseFfn;

  const eps = num('attention.layer_norm_rms_epsilon');
  if (eps !== null) config.rms_norm_eps = eps;

  const theta = num('rope.freq_base');
  if (theta !== null) { config.rope_theta_global = theta; config.rope_theta_swa = theta; }

  const maxCtx = maxContextLength ?? 2048;
  const ctxLen = num('context_length');
  if (ctxLen !== null) config.context_length = Math.min(ctxLen, maxCtx);

  const vocab = num('vocab_size');
  if (vocab !== null) config.vocab_size = vocab;

  // MoE geometry.
  const moe = config.moe!;
  const nExp = num('expert_count');
  if (nExp !== null) moe.num_experts = nExp;
  const usedExp = num('expert_used_count');
  if (usedExp !== null) moe.experts_per_tok = usedExp;
  const expFfn = num('expert_feed_forward_length');
  if (expFfn !== null) moe.moe_intermediate_size = expFfn;
  const sharedCount = num('expert_shared_count');
  if (sharedCount !== null) moe.shared_intermediate_size = sharedCount * moe.moe_intermediate_size;
  const denseLead = num('leading_dense_block_count');
  if (denseLead !== null) moe.first_dense_layers = denseLead;

  // Untied LM head (`output.weight` present in the Unlimited-OCR GGUF).
  const hasOutput = gguf.tensors.some((t) => t.name === 'output.weight');
  config.lm_head_tensor = hasOutput ? 'output' : undefined;

  // Recompute derived aggregates + per-layer arrays.
  config.q_dim = config.num_q_heads * config.head_dim;
  config.kv_dim = config.num_kv_heads * config.head_dim;
  config.attention_is_sliding = new Array<boolean>(config.num_layers).fill(false);
  config.intermediate_sizes = Array.from({ length: config.num_layers }, (_, i) =>
    i < moe.first_dense_layers ? config.intermediate_size : moe.shared_intermediate_size,
  );
  config.num_unshared_layers = config.num_layers;
  config.kv_producer_for_layer = Array.from({ length: config.num_layers }, (_, i) => i);

  return config;
}
