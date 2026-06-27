# Phase 0 — Qwen3 spike on the gemma4-webgpu engine

Forked from `tylerstraub/gemma4-webgpu` (Apache-2.0) on branch `qwen3-spike`.
Goal: prove the engine generalises to Qwen3 by loading a **Qwen3-4B Q4_K_M GGUF**,
decoding tokens, and passing `crossLabDiff` (cosine ≥0.94/layer, logits ≥0.9978)
against a Qwen3 PyTorch reference. Gate the whole Qwen3 kernel-pack effort on this.

## Why this base is the right call (verified 2026-06-27)

- **Already reads GGUF** (`src/gguf.ts`: F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K → F16).
  No loader to graft. Qwen3-4B GGUFs (Unsloth/llama.cpp) load by URL override.
- **F16-everywhere, one matmul path** — no quant-axis templating to port.
- **Per-head QK-norm already exists** (`per_head_rms_norm.wgsl`,
  `fused_per_head_norm_rope.wgsl`) — Qwen3's defining feature beyond llama.
- **`crossLabDiff` parity harness built in** (`docs/methodology.md`, `reference/pytorch`).
- Build green out of the box: `npm install` (26 pkgs), `npm run typecheck` passes.

**Qwen3 is a strict simplification of Gemma 4 + one MLP swap.**

## Architecture delta map (Gemma 4 E2B → Qwen3-4B)

| Aspect | Gemma 4 E2B (current) | Qwen3-4B | Action |
|---|---|---|---|
| Attention mask | sliding-window on 28/35 layers | full-causal everywhere | config: `attention_is_sliding` all `false` |
| RoPE | dual theta (global 1e6 / swa 1e4) | single theta 1e6 | all-global ⇒ only `rope_theta_global` used |
| Head dim | mixed 256/512 | uniform (128) | config: `head_dim_local == head_dim_global` |
| KV sharing | shared across layer groups (15 producers) | none | config: `num_unshared_layers = num_layers`, `kv_producer_for_layer[i]=i` |
| Per-layer embeddings (PLE) | yes (`ple_*.wgsl`) | none | gate off: `per_layer_input_dim = 0` + skip PLE stages |
| Final-logit softcap | tanh @30 (`logit_softcap.wgsl`) | none | gate off: `final_logit_softcapping = 0` |
| QK-norm | per-head RMSNorm | per-head RMSNorm (q_norm/k_norm) | **reuse** — confirm GGUF q/k_norm tensors wired |
| MLP activation | GELU (`gelu_mul.wgsl`) | **SwiGLU / SiLU** | **NEW**: add `silu_mul.wgsl` + config flag |
| GGUF metadata keys | `gemma4.*` | `qwen3.*` | map in a `configFromGGUF` for Qwen3 |

Most deltas are config "off" switches. The **two real code deltas**:
1. **SiLU MLP** — add `shaders/silu_mul.wgsl` (SiLU(gate)*up) + branch in the FFN
   path on an activation flag. (~1 shader + small engine wiring.)
2. **Qwen3 config + GGUF hydration** — `src/config/qwen3-4b.ts` mirroring
   `gemma4-e2b.ts`, reading `qwen3.*` metadata.

Plus: confirm PLE / softcap / dual-RoPE are **config-gated, not hardcoded**, in
`src/engine.ts` (2746 lines — the forward/layer loop). If hardcoded, gate them on
a model "kind" or the off-values above.

## Step plan

- [x] Clone + branch + install + typecheck green
- [x] Shader + config delta map (this doc)
- [ ] Read `src/engine.ts` forward pass; list every Gemma-specific that isn't
      already config-gated (PLE stages, softcap, dual-RoPE selection, GELU)
- [ ] `src/config/qwen3-4b.ts` (baseline + `configFromGGUF` for `qwen3.*`)
- [ ] `shaders/silu_mul.wgsl` + FFN activation branch
- [ ] Gate PLE / softcap off cleanly for the Qwen3 kind
- [ ] Obtain a Qwen3-4B Q4_K_M GGUF + generate a Qwen3 `smoke.npz` via
      `reference/pytorch` (mirror the Gemma reference generator)
- [ ] Load the GGUF, greedy-decode ~20 tokens
- [ ] `crossLabDiff` ≥ thresholds → **go/no-go**

## Risks specific to this base

- The engine's **F16-everywhere** choice means correctness depends on dequant +
  per-op f16 accumulation matching the reference — the parity harness is exactly
  for this; trust it, gate on it (CLAUDE.md: correctness before bench).
- Qwen3-4B GGUF QK-norm tensor names must map to what the per-head-norm path
  expects; verify tensor wiring during config hydration.
- `engine.ts` may bake Gemma assumptions (PLE always on, softcap always applied)
  into the hot path; budget the forward-pass read before writing the Qwen3 config.
