"""
Unlimited-OCR decoder reference generator for the engine's crossLabDiff sweep.

Loads ONLY the language-model half of baidu/Unlimited-OCR (the DeepSeek-V2 MoE
decoder — vision tensors are filtered out) from the locally-downloaded bf16
safetensors, runs one full-causal forward on a fixed RAW prompt, and dumps
per-layer residual hidden states + final-norm + logits at the LAST prompt
position to public/ref/deepseek_ocr_smoke.npz — the exact contract
qwen3_smoke.py established (see that file for the key-naming rationale).

Attention: use_mla=false in the HF config routes to SlidingWindowLlamaAttention
('mha_eager') — plain Llama MHA. The ring-buffer/R-SWA machinery only engages
when config._ring_window is set (generate-time); a bare forward is exactly
full-causal attention, which matches the engine's P0 scope.

Tokens: the GGUF carries add_bos_token=true, so the engine's encodeBpe prepends
BOS (id 0). We mirror that here: tokens = [bos] + encode(prompt, no specials).

Runs in the PINNED env (transformers 4.46.3 — the vendored modeling code needs
`_prepare_4d_causal_attention_mask`, removed in 5.x):

  /Users/chiragpatnaik/Code/browser-big-fast-lab/deepseek-ocr-spike/ref-env/bin/python \
      deepseek_ocr_smoke.py

Expected: ~1 min load + ~1-2 min CPU bf16 forward.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from deepseek_ocr_pkg.configuration_deepseek_v2 import DeepseekV2Config
from deepseek_ocr_pkg.modeling_deepseekv2 import DeepseekV2ForCausalLM

MODEL_DIR = Path("/Users/chiragpatnaik/Code/browser-big-fast-lab/deepseek-ocr-spike/models/hf")
PROMPT = "The capital of France is"
KEY_SUFFIX = "pos14"  # literal suffix the engine's capturePointToRefKey expects
OUT = Path(__file__).resolve().parents[2] / "public" / "ref"
OUT.mkdir(parents=True, exist_ok=True)

# Vision / multimodal tensor prefixes to drop from the checkpoint.
VISION_PREFIXES = (
    "model.sam_model.", "model.vision_model.", "model.projector.",
    "model.image_newline", "model.view_seperator",
)


def main():
    t0 = time.time()
    full_cfg = json.loads((MODEL_DIR.parent.parent / "upstream" / "config.json").read_text())
    lang_cfg = dict(full_cfg["language_config"])
    # Drop HF-plumbing keys that DeepseekV2Config doesn't take positionally;
    # everything else (hidden sizes, MoE geometry, use_mla=false, v_head_dim,
    # sliding_window_size…) passes through as kwargs.
    for k in ("architectures", "auto_map", "torch_dtype"):
        lang_cfg.pop(k, None)
    config = DeepseekV2Config(**lang_cfg)
    config._attn_implementation = "eager"   # → ATTENTION_CLASSES['mha_eager']
    assert not config.use_mla, "expected use_mla=false (plain Llama MHA)"
    print(f"[{time.time()-t0:6.1f}s] Config: layers={config.num_hidden_layers} hidden={config.hidden_size} "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads} "
          f"experts={config.n_routed_experts}(+{config.n_shared_experts} shared) top{config.num_experts_per_tok} "
          f"norm_topk_prob={config.norm_topk_prob} scale={config.routed_scaling_factor} "
          f"rope_theta={config.rope_theta} first_dense={config.first_k_dense_replace}")

    print(f"[{time.time()-t0:6.1f}s] Instantiating decoder (bf16 init)…")
    torch.set_default_dtype(torch.bfloat16)
    model = DeepseekV2ForCausalLM(config)
    torch.set_default_dtype(torch.float32)
    model.eval()

    print(f"[{time.time()-t0:6.1f}s] Loading safetensors (6.67 GB)…")
    sd = load_file(str(MODEL_DIR / "model-00001-of-000001.safetensors"))
    lang_sd = {k: v for k, v in sd.items() if not k.startswith(VISION_PREFIXES)}
    print(f"[{time.time()-t0:6.1f}s] {len(sd)} tensors → {len(lang_sd)} language tensors")
    missing, unexpected = model.load_state_dict(lang_sd, strict=False)
    # Non-persistent buffers (rotary inv_freq) are expected in `missing`.
    real_missing = [k for k in missing if "inv_freq" not in k and "rotary" not in k]
    print(f"[{time.time()-t0:6.1f}s] load_state_dict: missing={len(missing)} "
          f"(real: {real_missing[:8]}) unexpected={len(unexpected)} ({unexpected[:4]})")
    assert not real_missing, f"missing non-buffer weights: {real_missing[:20]}"
    del sd, lang_sd

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=False)
    raw = tok(PROMPT, add_special_tokens=False)["input_ids"]
    bos = config.bos_token_id
    tokens = [bos] + raw
    last_pos = len(tokens) - 1
    print(f"[{time.time()-t0:6.1f}s] Prompt {PROMPT!r} → bos({bos}) + {raw} = {tokens} (capture pos={last_pos})")

    inner = model.model
    lm_head = model.lm_head
    n_layers = config.num_hidden_layers

    layer_resid: dict[int, torch.Tensor] = {}

    def mk_hook(idx):
        def hook(_mod, _inp, out):
            layer_resid[idx] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [inner.layers[i].register_forward_hook(mk_hook(i)) for i in range(n_layers)]

    input_ids = torch.tensor([tokens], dtype=torch.long)
    print(f"[{time.time()-t0:6.1f}s] Forward pass (CPU, bf16, full-causal — no ring window)…")
    with torch.no_grad():
        inner(input_ids=input_ids, use_cache=False, output_hidden_states=False)
        final_resid = layer_resid[n_layers - 1]
        final_normed = inner.norm(final_resid)
        logits = lm_head(final_normed)
    for h in handles:
        h.remove()
    print(f"[{time.time()-t0:6.1f}s] Forward complete. logits shape={tuple(logits.shape)}")

    with torch.no_grad():
        embed = inner.embed_tokens(input_ids)  # no embedding scale in DeepseekV2

    last_logits = logits[0, last_pos].float()
    topv, topi = torch.topk(last_logits, k=8)
    print("top-8 next-token (id, logit, decoded):")
    for v, i in zip(topv.tolist(), topi.tolist()):
        dec = tok.decode([i]).encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"  {i:6d}  {v:+8.4f}  {dec!r}")

    dump = {
        "tokens": np.array(tokens, dtype=np.int32),
        "position": np.int32(last_pos),
        f"embed_{KEY_SUFFIX}": embed[0, last_pos].float().cpu().numpy(),
        f"final_{KEY_SUFFIX}": final_normed[0, last_pos].float().cpu().numpy(),
        f"logits_{KEY_SUFFIX}": last_logits.cpu().numpy(),
    }
    for L in range(n_layers):
        dump[f"afterLayer_{L:02d}_{KEY_SUFFIX}"] = layer_resid[L][0, last_pos].float().cpu().numpy()

    out_path = OUT / "deepseek_ocr_smoke.npz"
    np.savez(out_path, **dump)
    print(f"[{time.time()-t0:6.1f}s] Dumped {len(dump)} arrays → {out_path}")

    manifest = {
        "model": "baidu/Unlimited-OCR (language decoder only)",
        "prompt": PROMPT,
        "tokens": tokens,
        "position": last_pos,
        "num_layers": n_layers,
        "keys": sorted(dump.keys()),
        "note": "afterLayer_LL = residual after layer LL (pre-final-norm); final = post-final-norm; "
                "logits = lm_head(final), no softcap. Full-causal (no R-SWA ring) — matches engine P0. "
                "bf16 CPU forward, transformers 4.46.3, vendored Baidu modeling code (mha_eager).",
    }
    (OUT / "deepseek_ocr_smoke_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{time.time()-t0:6.1f}s] Wrote manifest. tokens for engine cross-check: {tokens}")


if __name__ == "__main__":
    main()
