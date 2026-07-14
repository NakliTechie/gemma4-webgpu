#!/usr/bin/env python3
"""Reference side of the Q5_0 dequant cross-check (see check_q5_0.mjs).

    python3 scripts/check_q5_0.py <model.gguf> <tensor-name> [nElems]

Uses the `gguf` package's GGUFReader + dequantize — the llama.cpp-blessed
reference implementation. Prints one value per line.
"""
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

file, name = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 64

reader = GGUFReader(file)
t = next(t for t in reader.tensors if t.name == name)
print(f"tensor {t.name} type={t.tensor_type} shape={t.shape}", file=sys.stderr)

vals = dequantize(t.data, t.tensor_type).flatten()
for v in vals[:n]:
    print(f"{v:.8g}")
