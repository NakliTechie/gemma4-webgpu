"""
DeepEncoder → ONNX export (P1 of the Unlimited-OCR port).

Exports the full vision stack of baidu/Unlimited-OCR — SAM ViT-B (window-14
attention, decomposed rel-pos, 16× conv compressor) → CLIP-L (fed the SAM
features as patch_embeds) → concat fusion → linear projector — as ONE graph:

    pixel_values [1, 3, 1024, 1024] f32 (normalized, mean/std 0.5)
      → vision_embeds [1, 256, 1280] f32

The decoder consumes vision_embeds via the engine's (P1) inputs_embeds path;
`image_newline` / `view_seperator` embeddings are spliced JS-side, so they are
NOT part of this graph (dumped to a sidecar npz instead).

fp32 first — parity vs torch is the gate; fp16/quant is a later size pass.
Static 1024² shape (the model's native `candidate_resolutions` and the P0/P1
single-tile scope). Opset 18 (antialias bicubic Resize needs ≥18 — CLIP pos-
embed interpolation).

Run in the pinned env:
  /Users/chiragpatnaik/Code/browser-big-fast-lab/deepseek-ocr-spike/ref-env/bin/python \
      export_deepencoder_onnx.py
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from deepseek_ocr_pkg.deepencoder import build_sam_vit_b, build_clip_l, MlpProjector
from easydict import EasyDict

MODEL_DIR = Path("/Users/chiragpatnaik/Code/browser-big-fast-lab/deepseek-ocr-spike/models/hf")
OUT_DIR = Path("/Users/chiragpatnaik/Code/browser-big-fast-lab/deepseek-ocr-spike/models/onnx")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class DeepEncoder(nn.Module):
    """SAM → CLIP(pixels, sam_feats) → concat → linear projector.

    Mirrors modeling_unlimitedocr.py's single-view path exactly:
        f1 = sam_model(image)                                   # [B,1024,16,16]
        f2 = vision_model(image, f1)                            # [B,257,1024]
        feat = cat((f2[:,1:], f1.flatten(2).permute(0,2,1)), -1)# [B,256,2048]
        out = projector(feat)                                   # [B,256,1280]
    """

    def __init__(self):
        super().__init__()
        self.sam_model = build_sam_vit_b()
        self.vision_model = build_clip_l()
        self.projector = MlpProjector(EasyDict(projector_type="linear", input_dim=2048, n_embed=1280))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        f1 = self.sam_model(pixel_values)
        f2 = self.vision_model(pixel_values, f1)
        feat = torch.cat((f2[:, 1:], f1.flatten(2).permute(0, 2, 1)), dim=-1)
        return self.projector(feat)


def main():
    t0 = time.time()
    enc = DeepEncoder().eval()

    print(f"[{time.time()-t0:6.1f}s] Loading vision tensors…")
    sd = load_file(str(MODEL_DIR / "model-00001-of-000001.safetensors"))
    sub = {}
    extras = {}
    for k, v in sd.items():
        if k.startswith("model.sam_model."):
            sub["sam_model." + k[len("model.sam_model."):]] = v.float()
        elif k.startswith("model.vision_model."):
            sub["vision_model." + k[len("model.vision_model."):]] = v.float()
        elif k.startswith("model.projector."):
            sub["projector." + k[len("model.projector."):]] = v.float()
        elif k in ("model.image_newline", "model.view_seperator"):
            extras[k.split(".")[-1]] = v.float().numpy()
    del sd
    missing, unexpected = enc.load_state_dict(sub, strict=False)
    print(f"[{time.time()-t0:6.1f}s] vision tensors={len(sub)} missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:10])
    if unexpected:
        print("  unexpected:", unexpected[:10])
    real_missing = [k for k in missing if "position_ids" not in k]  # arange buffer, not a weight
    assert not real_missing, f"missing vision weights: {real_missing[:10]}"

    # Splice embeddings the decoder-side JS needs (not part of the graph).
    np.savez(OUT_DIR / "deepencoder_extras.npz", **extras)
    print(f"[{time.time()-t0:6.1f}s] extras: {list(extras)} → deepencoder_extras.npz")

    # Torch reference forward on a deterministic pseudo-image.
    rng = np.random.default_rng(42)
    px = rng.standard_normal((1, 3, 1024, 1024), dtype=np.float32) * 0.5
    x = torch.from_numpy(px)
    with torch.no_grad():
        ref = enc(x)
    print(f"[{time.time()-t0:6.1f}s] torch forward: out={tuple(ref.shape)} norm={ref.norm():.4f}")
    np.save(OUT_DIR / "deepencoder_ref_in.npy", px)
    np.save(OUT_DIR / "deepencoder_ref_out.npy", ref.numpy())

    onnx_path = OUT_DIR / "deepencoder_fp32.onnx"
    print(f"[{time.time()-t0:6.1f}s] Exporting ONNX (opset 18, static 1024²)…")
    torch.onnx.export(
        enc,
        (x,),
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["vision_embeds"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    size_mb = onnx_path.stat().st_size / 1e6
    ext = list(OUT_DIR.glob("deepencoder_fp32.onnx.data")) + list(OUT_DIR.glob("*.onnx_data"))
    print(f"[{time.time()-t0:6.1f}s] Exported {onnx_path.name} ({size_mb:.1f} MB + ext {[e.name for e in ext]})")

    # Parity: python onnxruntime CPU.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (out,) = sess.run(None, {"pixel_values": px})
    ref_np = ref.numpy()
    cos = float(np.dot(out.ravel(), ref_np.ravel()) / (np.linalg.norm(out) * np.linalg.norm(ref_np)))
    max_abs = float(np.abs(out - ref_np).max())
    print(f"[{time.time()-t0:6.1f}s] ORT parity: cosine={cos:.7f} maxAbsDiff={max_abs:.5f}")
    verdict = "PARITY OK" if cos > 0.9999 else "PARITY FAIL"
    print(f"[{time.time()-t0:6.1f}s] {verdict}")


if __name__ == "__main__":
    main()
