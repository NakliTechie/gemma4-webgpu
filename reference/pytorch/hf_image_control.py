"""Ground-truth control for the browser e2e: run the SAME document through the
full HF stack (DeepEncoder torch fp32 + bf16 decoder), global-view-only
(no crops — mirrors the browser chain exactly), prompt '<image>\\nFree OCR.',
greedy. If THIS emits EOS immediately too, the browser chain is faithful and
the prompt/format needs work; if it emits text, the browser chain has a bug.

Also dumps the vision embeds + spliced sequence so the browser side can be
diffed elementwise later.
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
OUT_DIR = Path("/Users/chiragpatnaik/Code/browser-big-fast-lab/deepseek-ocr-spike/models/onnx")
t0 = time.time()

# ── document (same text as the browser canvas) ──
img = Image.new("RGB", (1024, 1024), "white")
d = ImageDraw.Draw(img)
def font(size, bold=False):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Georgia%s.ttf" % (" Bold" if bold else ""), size)
    except OSError:
        return ImageFont.load_default(size)
d.text((64, 70), "Quarterly Report", font=font(56, True), fill="black")
d.text((64, 190), "Revenue grew 18% year over year.", font=font(34), fill="black")
d.text((64, 250), "Operating costs fell by 4%.", font=font(34), fill="black")
d.text((64, 310), "The board approved the 2027 plan.", font=font(34), fill="black")
d.text((64, 400), "Prepared by the finance team.", font=font(28), fill="black")
img.save(OUT_DIR / "control_doc.png")

px = (np.asarray(img, dtype=np.float32) / 255.0 - 0.5) / 0.5      # HWC
x = torch.from_numpy(px.transpose(2, 0, 1))[None]                  # [1,3,1024,1024]

# ── vision (fp32) ──
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
with torch.no_grad():
    patches = enc(x)[0]                       # [256,1280]
print(f"[{time.time()-t0:.1f}s] vision: {tuple(patches.shape)} rowNorm mean={patches.norm(dim=-1).mean():.3f}")

grid = patches.view(16, 16, 1280)
rows = torch.cat([grid, newline[None, None, :].expand(16, 1, 1280)], dim=1).reshape(-1, 1280)
vision_seq = torch.cat([rows, seperator[None, :]], dim=0)          # [273,1280]
np.save(OUT_DIR / "control_vision_seq.npy", vision_seq.numpy())

# ── decoder (bf16) ──
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
import sys
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "\nFree OCR."
prompt_ids = tok(PROMPT, add_special_tokens=False)["input_ids"]
print(f"prompt: {PROMPT!r} → {prompt_ids}")
ids = [BOS] + [128815] * 273 + prompt_ids
embeds = model.model.embed_tokens(torch.tensor([ids]))            # [1,T,1280] bf16
embeds[0, 1:274] = vision_seq.to(torch.bfloat16)
print(f"[{time.time()-t0:.1f}s] seq len={len(ids)} — greedy decode…")

past = None
generated = []
cur = embeds
with torch.no_grad():
    for step in range(80):
        out = model(inputs_embeds=cur, use_cache=True, past_key_values=past)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())
        if step == 0:
            v, i = torch.topk(out.logits[0, -1].float(), 8)
            print("  first-step top-8:", [(int(a), round(float(b), 2), tok.decode([int(a)])) for b, a in zip(v, i)])
        if nxt == EOS:
            print(f"  EOS at step {step}")
            break
        generated.append(nxt)
        cur = model.model.embed_tokens(torch.tensor([[nxt]]))
print(f"[{time.time()-t0:.1f}s] HF control output ({len(generated)} tokens):")
print(repr(tok.decode(generated)))
