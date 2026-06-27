"""
Qwen3-1.7B reference generator for the engine's crossLabDiff sweep.

Loads Qwen/Qwen3-1.7B (bf16 safetensors), runs one forward pass on a fixed RAW
prompt (no chat template → tokenization matches the engine's GGUF tokenizer
exactly), and dumps per-layer residual-stream hidden states + final-norm + logits
at the LAST prompt position, to public/ref/qwen3_smoke.npz.

Reference semantics are matched to the engine's CapturePoint contract:
  - embed_pos14      = token embeddings (Qwen3 does NOT scale embeddings)
  - afterLayer_LL_pos14 = residual stream AFTER decoder layer LL (PRE final norm)
  - final_pos14      = post-final-RMSNorm hidden state
  - logits_pos14     = lm_head(final)  (Qwen3 has no logit softcap)

The "_pos14" suffix is the key name the engine's capturePointToRefKey() expects;
the actual numeric position is len(tokens)-1 (printed below). Both sides capture
the same last token at the same position, so the label is cosmetic.

Run:  uv run python qwen3_smoke.py
Expected: ~30s load + ~1-2 min CPU forward for a short prompt.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
PROMPT = "The capital of France is"
# Engine ref keys are hardcoded to the "_pos14" suffix (Gemma's 15-token canonical).
# We reuse that literal suffix regardless of this prompt's actual length.
KEY_SUFFIX = "pos14"
OUT = Path(__file__).resolve().parents[2] / "public" / "ref"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading {MODEL_ID} (bf16, CPU)…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",  # deterministic, hook-friendly
        low_cpu_mem_usage=True,
    )
    model.eval()
    inner = model.model            # Qwen3Model (decoder stack + embed + final norm)
    lm_head = model.lm_head
    n_layers = model.config.num_hidden_layers
    print(f"[{time.time()-t0:6.1f}s] Loaded. layers={n_layers} hidden={model.config.hidden_size} "
          f"heads={model.config.num_attention_heads}/{model.config.num_key_value_heads} "
          f"head_dim={getattr(model.config,'head_dim','?')} params={sum(p.numel() for p in model.parameters())/1e9:.2f}B")

    # RAW tokenization (no chat template, no special tokens) so the engine's
    # encodePromptTokens(prompt, applyChatTemplate=false) yields identical ids.
    enc = tok(PROMPT, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    tokens = input_ids[0].tolist()
    last_pos = len(tokens) - 1
    print(f"[{time.time()-t0:6.1f}s] Prompt {PROMPT!r} → {len(tokens)} tokens: {tokens}  (capture pos={last_pos})")

    # Capture each decoder layer's OUTPUT residual (pre-final-norm) via hooks —
    # this matches the engine's afterLayer:L semantics exactly.
    layer_resid: dict[int, torch.Tensor] = {}

    def mk_hook(idx):
        def hook(_mod, _inp, out):
            # Qwen3DecoderLayer returns a tuple; out[0] is the hidden_states residual.
            layer_resid[idx] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [inner.layers[i].register_forward_hook(mk_hook(i)) for i in range(n_layers)]

    print(f"[{time.time()-t0:6.1f}s] Forward pass (CPU, bf16)…")
    with torch.no_grad():
        out = inner(input_ids=input_ids, use_cache=False, output_hidden_states=False)
        last_hidden = out.last_hidden_state           # already post-final-norm in Qwen3Model
        # Recompute final-norm explicitly from the last layer residual to be unambiguous.
        final_resid = layer_resid[n_layers - 1]
        final_normed = inner.norm(final_resid)
        logits = lm_head(final_normed)                # [1, T, vocab]; Qwen3 has no softcap
    for h in handles:
        h.remove()
    print(f"[{time.time()-t0:6.1f}s] Forward complete. logits shape={tuple(logits.shape)}")

    # Embedding (post-embed, pre-layer-0). Qwen3 does not scale embeddings.
    with torch.no_grad():
        embed = inner.embed_tokens(input_ids)

    # Sanity: top token at the last position.
    last_logits = logits[0, last_pos].float()
    topv, topi = torch.topk(last_logits, k=8)
    print("top-8 next-token (id, logit, decoded):")
    for v, i in zip(topv.tolist(), topi.tolist()):
        dec = tok.decode([i]).encode("ascii", errors="backslashreplace").decode("ascii")
        print(f"  {i:6d}  {v:+8.4f}  {dec!r}")

    # Dump F32 arrays at the last position.
    dump = {
        "tokens": np.array(tokens, dtype=np.int32),
        "position": np.int32(last_pos),
        f"embed_{KEY_SUFFIX}": embed[0, last_pos].float().cpu().numpy(),
        f"final_{KEY_SUFFIX}": final_normed[0, last_pos].float().cpu().numpy(),
        f"logits_{KEY_SUFFIX}": last_logits.cpu().numpy(),
    }
    for L in range(n_layers):
        dump[f"afterLayer_{L:02d}_{KEY_SUFFIX}"] = layer_resid[L][0, last_pos].float().cpu().numpy()

    out_path = OUT / "qwen3_smoke.npz"
    np.savez(out_path, **dump)
    print(f"[{time.time()-t0:6.1f}s] Dumped {len(dump)} arrays → {out_path}")

    manifest = {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "tokens": tokens,
        "position": last_pos,
        "num_layers": n_layers,
        "keys": sorted(dump.keys()),
        "note": "afterLayer_LL = residual after layer LL (pre-final-norm); final = post-final-norm; logits = lm_head(final). No softcap (Qwen3).",
    }
    (OUT / "qwen3_smoke_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[{time.time()-t0:6.1f}s] Wrote manifest. tokens for engine cross-check: {tokens}")


if __name__ == "__main__":
    main()
