"""fp16 pass on the DeepEncoder ONNX (P2 item 5): 1.6 GB fp32 → ~800 MB fp16.

Two-attempt policy per the lab's fp16-converter history (RMSNorm corruption on
the Qwen3 side; vision here is LayerNorm — must verify, not assume):
  attempt 1: plain convert_float_to_float16(keep_io_types=True)
  attempt 2: + op_block_list for normalization/softmax/resize ops kept fp32
Gate: ORT parity vs the fp32 graph on BOTH the seed-42 fixture and the real
control document. Accept if cosine ≥ 0.999 and the e2e-relevant row norms hold.
"""
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxconverter_common import float16

OUT_DIR = Path("/Users/chiragpatnaik/Code/browser-big-fast-lab/deepseek-ocr-spike/models/onnx")
FP32 = OUT_DIR / "deepencoder_fp32.onnx"
FP16 = OUT_DIR / "deepencoder_fp16.onnx"
t0 = time.time()

ref_in = np.load(OUT_DIR / "deepencoder_ref_in.npy")
ref_out = np.load(OUT_DIR / "deepencoder_ref_out.npy")

from PIL import Image
doc = np.asarray(Image.open(OUT_DIR / "control_doc.png"), dtype=np.float32)
doc_px = ((doc / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]

def parity(path, tag):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (o1,) = sess.run(None, {"pixel_values": ref_in})
    cos1 = float(np.dot(o1.ravel(), ref_out.ravel()) / (np.linalg.norm(o1) * np.linalg.norm(ref_out)))
    (o2,) = sess.run(None, {"pixel_values": doc_px})
    return cos1, o2

print(f"[{time.time()-t0:.1f}s] fp32 baseline on control doc…")
sess32 = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
(doc32,) = sess32.run(None, {"pixel_values": doc_px})
del sess32

def attempt1():
    # Plain converter. Known failure mode on this graph: pre-existing Cast
    # nodes in SAM rel-pos attention end up type-inconsistent (observed:
    # session load fails on /sam_model/blocks.0/attn/Cast_14).
    model = onnx.load(str(FP32))
    m16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(m16, str(FP16))

def attempt2():
    # auto_convert_mixed_precision: converts greedily WHILE validating outputs
    # against the fp32 graph on a real feed — handles Cast chains and keeps
    # numerically-sensitive subgraphs (norms etc.) fp32 automatically.
    from onnxconverter_common import auto_mixed_precision
    model = onnx.load(str(FP32))
    m16 = auto_mixed_precision.auto_convert_mixed_precision(
        model, {"pixel_values": ref_in}, rtol=0.01, keep_io_types=True)
    onnx.save(m16, str(FP16))

ok = False
for attempt, fn in ((1, attempt1), (2, attempt2)):
    print(f"[{time.time()-t0:.1f}s] attempt {attempt} ({fn.__name__})…")
    try:
        fn()
        size_mb = FP16.stat().st_size / 1e6
        cos_fixture, doc16 = parity(FP16, "fp16")
        cos_doc = float(np.dot(doc16.ravel(), doc32.ravel()) / (np.linalg.norm(doc16) * np.linalg.norm(doc32)))
        max_abs = float(np.abs(doc16 - doc32).max())
        print(f"  size={size_mb:.0f} MB · cos(fixture vs torch-fp32)={cos_fixture:.6f} · cos(doc vs onnx-fp32)={cos_doc:.6f} · maxAbs={max_abs:.4f}")
        if cos_fixture >= 0.999 and cos_doc >= 0.999:
            print(f"[{time.time()-t0:.1f}s] PARITY OK on attempt {attempt} — {FP16.name} accepted")
            ok = True
            break
    except Exception as e:
        print(f"  attempt {attempt} failed: {type(e).__name__}: {str(e)[:300]}")
if not ok:
    FP16.unlink(missing_ok=True)
    print(f"[{time.time()-t0:.1f}s] PARITY FAIL after 2 attempts — fp16 rejected, fp32 stays canonical")
