"""Tier-2 ground truth for the R-SWA ring: run the full HF bf16 stack on a
3-page document with the vendored ring ENGAGED (config._ring_window = 128,
exactly as modeling_unlimitedocr.infer_multi does), decode well past the wrap
point, and save everything the browser side needs to reproduce the run
bit-comparably: page PNGs, prompt ids, HF ring tokens/text, HF plain
tokens/text (the ring-vs-plain divergence is itself evidence the reference
ring engaged).

Multi-page layout (verified from infer_multi): per page
  ([id]*16 + [id]) * 16 + [id]  = 273 placeholder tokens
(the trailing token doubles as the between-pages separator), embeds side =
16 rows × (16 patches + image_newline) + view_seperator = 273. Sequence:
[BOS] + N×273 + tokenize("Multi page parsing.").
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file
from transformers import AutoTokenizer
from easydict import EasyDict

from deepseek_ocr_pkg.configuration_deepseek_v2 import DeepseekV2Config
from deepseek_ocr_pkg.modeling_deepseekv2 import DeepseekV2ForCausalLM
from deepseek_ocr_pkg.deepencoder import build_sam_vit_b, build_clip_l, MlpProjector

MODEL_DIR = Path("/Users/chiragpatnaik/Code/browser-big-fast-lab/deepseek-ocr-spike/models/hf")
OUT_DIR = Path("/Users/chiragpatnaik/Code/browser-big-fast-lab/deepseek-ocr-spike/models/longdoc")
OUT_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.time()

PAGES = [
    ("Field Report — Page 1", [
        "The northern survey began on March 4th.",
        "Twelve stations reported stable readings.",
        "Sensor B7 required a manual recalibration.",
        "Water levels rose two centimeters overnight.",
        "The team logged fourteen soil samples.",
        "Supplies remain sufficient for nine days.",
    ]),
    ("Field Report — Page 2", [
        "On March 6th the wind shifted southeast.",
        "Station C2 recorded the lowest pressure.",
        "Two drones mapped the eastern ridge line.",
        "A minor rockslide blocked the access trail.",
        "Repairs to the trail took most of the day.",
        "Morale stayed high despite the setback.",
    ]),
    ("Field Report — Page 3", [
        "By March 9th all stations were nominal.",
        "The aquifer model matched observed data.",
        "Final samples were sealed and labeled.",
        "The convoy departs at dawn on March 11th.",
        "Archive copies were sent to the base camp.",
        "This concludes the northern survey report.",
    ]),
]

def font(size, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia%s.ttf" % (" Bold" if bold else ""), size)
    except OSError:
        return ImageFont.load_default(size)

pixel_pages = []
for i, (title, lines) in enumerate(PAGES):
    img = Image.new("RGB", (1024, 1024), "white")
    d = ImageDraw.Draw(img)
    d.text((64, 60), title, font=font(48, True), fill="black")
    for j, line in enumerate(lines):
        d.text((64, 180 + j * 64), line, font=font(32), fill="black")
    d.text((64, 940), f"- {i+1} -", font=font(26), fill="black")
    img.save(OUT_DIR / f"page_{i+1}.png")
    px = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    pixel_pages.append(torch.from_numpy(px.transpose(2, 0, 1))[None])
print(f"[{time.time()-t0:.1f}s] {len(PAGES)} pages rendered")

# ── vision ──
class DeepEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        self.projector = MlpProjector(EasyDict(projector_type="linear", input_dim=2048, n_embed=1280))
    def forward(self, pv):
        f1 = self.sam_model(pv)
        f2 = self.vision_model(pv, f1)
        return self.projector(torch.cat((f2[:, 1:], f1.flatten(2).permute(0, 2, 1)), dim=-1))

enc = DeepEncoder().eval()
sd = load_file(str(MODEL_DIR / "model-00001-of-000001.safetensors"))
enc.load_state_dict({k[len("model."):]: v.float() for k, v in sd.items()
                     if k.startswith(("model.sam_model.", "model.vision_model.", "model.projector."))}, strict=False)
newline = sd["model.image_newline"].float()
seperator = sd["model.view_seperator"].float()

blocks = []
for i, x in enumerate(pixel_pages):
    with torch.no_grad():
        patches = enc(x)[0]                                    # [256,1280]
    grid = patches.view(16, 16, 1280)
    rows = torch.cat([grid, newline[None, None, :].expand(16, 1, 1280)], dim=1).reshape(-1, 1280)
    blocks.append(torch.cat([rows, seperator[None, :]], dim=0))  # [273,1280]
    print(f"[{time.time()-t0:.1f}s] page {i+1} vision done")
vision_seq = torch.cat(blocks, dim=0)                          # [N*273,1280]

# ── decoder ──
cfg = json.loads((MODEL_DIR.parent.parent / "upstream" / "config.json").read_text())["language_config"]
for k in ("architectures", "auto_map", "torch_dtype"):
    cfg.pop(k, None)
config = DeepseekV2Config(**cfg)
config._attn_implementation = "eager"
torch.set_default_dtype(torch.bfloat16)
model = DeepseekV2ForCausalLM(config)
torch.set_default_dtype(torch.float32)
model.eval()
model.load_state_dict({k: v for k, v in sd.items() if not k.startswith(
    ("model.sam_model.", "model.vision_model.", "model.projector.", "model.image_newline", "model.view_seperator"))}, strict=False)
del sd
print(f"[{time.time()-t0:.1f}s] decoder loaded")

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
BOS, EOS = 0, 1
prompt_ids = tok("Multi page parsing.", add_special_tokens=False)["input_ids"]
print(f"prompt ids: {prompt_ids}")
n_img = vision_seq.shape[0]
ids = [BOS] + [128815] * n_img + prompt_ids

def greedy(ring: bool, max_new: int):
    if ring:
        model.config._ring_window = 128
        model.config.sliding_window = None
    else:
        model.config._ring_window = None
    embeds = model.model.embed_tokens(torch.tensor([ids]))
    embeds[0, 1:1 + n_img] = vision_seq.to(torch.bfloat16)
    past, cur, out = None, embeds, []
    with torch.no_grad():
        for step in range(max_new):
            o = model(inputs_embeds=cur, use_cache=True, past_key_values=past)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            if nxt == EOS:
                break
            out.append(nxt)
            cur = model.model.embed_tokens(torch.tensor([[nxt]]))
    return out

print(f"[{time.time()-t0:.1f}s] seq len={len(ids)} (P={len(ids)-len(prompt_ids)+len(prompt_ids)}) — RING greedy…")
ring_ids = greedy(True, 480)
ring_text = tok.decode(ring_ids)
print(f"[{time.time()-t0:.1f}s] ring: {len(ring_ids)} tokens")
print(f"[{time.time()-t0:.1f}s] PLAIN greedy…")
plain_ids = greedy(False, 480)
plain_text = tok.decode(plain_ids)
print(f"[{time.time()-t0:.1f}s] plain: {len(plain_ids)} tokens")

first_div = next((i for i in range(min(len(ring_ids), len(plain_ids))) if ring_ids[i] != plain_ids[i]), -1)
print(f"ring-vs-plain first divergence at generated step: {first_div} (must be ≥128 or -1)")
print("RING TEXT:")
print(ring_text)

json.dump({
    "prompt_ids": prompt_ids, "n_img_tokens": n_img, "seq_prefix_len": len(ids),
    "ring_ids": ring_ids, "ring_text": ring_text,
    "plain_ids": plain_ids, "plain_text": plain_text,
    "first_divergence": first_div,
    "pages": [t for t, _ in PAGES],
}, open(OUT_DIR / "hf_ring_control.json", "w"), indent=1)
np.save(OUT_DIR / "vision_seq_fp32.npy", vision_seq.numpy())
print(f"[{time.time()-t0:.1f}s] saved → {OUT_DIR}")
