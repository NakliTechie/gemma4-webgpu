//#endregion
//#region src/shaders.ts
var e = {
	add: "@group(0) @binding(0) var<storage, read> a: array<f32>;\n@group(0) @binding(1) var<storage, read> b: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> size: u32;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x;\n  if (i >= size) { return; }\n  output[i] = a[i] + b[i];\n}\n",
	argmax: "@group(0) @binding(0) var<storage, read> logits: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<u32>;\n@group(0) @binding(2) var<uniform> size: u32;\nvar<workgroup> shared_max: array<f32, 256>;\nvar<workgroup> shared_idx: array<u32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n  let tid = lid.x;\n  var local_max: f32 = -1e30;\n  var local_idx: u32 = 0u;\n  var i = tid;\n  while (i < size) {\n    let val = logits[i];\n    if (val > local_max) {\n      local_max = val;\n      local_idx = i;\n    }\n    i += 256u;\n  }\n  shared_max[tid] = local_max;\n  shared_idx[tid] = local_idx;\n  workgroupBarrier();\n  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {\n    if (tid < stride && shared_max[tid + stride] > shared_max[tid]) {\n      shared_max[tid] = shared_max[tid + stride];\n      shared_idx[tid] = shared_idx[tid + stride];\n    }\n    workgroupBarrier();\n  }\n  if (tid == 0u) {\n    result[0] = shared_idx[0];\n  }\n}\n",
	attnOutput: "struct Params { num_q_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32 }\n@group(0) @binding(0) var<storage, read> probs: array<f32>;\n@group(0) @binding(1) var<storage, read> v_cache: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let idx = gid.x;\n  let total = params.num_q_heads * params.head_dim;\n  if (idx >= total) { return; }\n  let head = idx / params.head_dim;\n  let d = idx % params.head_dim;\n  let kv_head = head * params.num_kv_heads / params.num_q_heads;\n  let kv_stride = params.num_kv_heads * params.head_dim;\n  var sum: f32 = 0.0;\n  for (var pos: u32 = 0u; pos < params.seq_len; pos++) {\n    let prob = probs[head * params.seq_len + pos];\n    let v_idx = pos * kv_stride + kv_head * params.head_dim + d;\n    sum += prob * v_cache[v_idx];\n  }\n  output[head * params.head_dim + d] = sum;\n}\n",
	attnScore: "// `sliding_window = 0` means full attention. A positive value W means\n// Gemma-4-style sliding window: position p only attends to keys in\n// `[current_pos - W + 1, current_pos]`. Masked positions get -inf so\n// softmax drops them. `current_pos = seq_len - 1`.\n//\n// `scale` is always 1.0 for Gemma 4 (q_norm + k_norm already normalize\n// each head to unit RMS, so the standard 1/sqrt(head_dim) compensation\n// is dropped — baking it in would flatten the softmax by sqrt(HD).)\nstruct Params { num_q_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32, scale: f32, sliding_window: u32 }\n@group(0) @binding(0) var<storage, read> q: array<f32>;\n@group(0) @binding(1) var<storage, read> k_cache: array<f32>;\n@group(0) @binding(2) var<storage, read_write> scores: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let idx = gid.x;\n  let total = params.num_q_heads * params.seq_len;\n  if (idx >= total) { return; }\n  let head = idx / params.seq_len;\n  let pos = idx % params.seq_len;\n  let score_idx = head * params.seq_len + pos;\n  if (params.sliding_window != 0u) {\n    let current_pos = params.seq_len - 1u;\n    let window_start = select(0u, current_pos + 1u - params.sliding_window, current_pos + 1u > params.sliding_window);\n    if (pos < window_start) {\n      scores[score_idx] = -1e30;\n      return;\n    }\n  }\n  let kv_head = head * params.num_kv_heads / params.num_q_heads;\n  let q_offset = head * params.head_dim;\n  let kv_stride = params.num_kv_heads * params.head_dim;\n  let k_offset = pos * kv_stride + kv_head * params.head_dim;\n  var dot: f32 = 0.0;\n  for (var d: u32 = 0u; d < params.head_dim; d++) {\n    dot += q[q_offset + d] * k_cache[k_offset + d];\n  }\n  scores[score_idx] = dot * params.scale;\n}\n",
	embeddingLookup: "enable f16;\n// embed_scale: Gemma 4 uses sqrt(hidden_size); Qwen3 uses 1.0 (no scaling).\n// Resolved on CPU and passed in so the kernel stays arch-agnostic.\nstruct Params { hidden_size: u32, token_id: u32, embed_scale: f32 }\n@group(0) @binding(0) var<storage, read> embedding: array<f16>;\n@group(0) @binding(1) var<storage, read_write> output: array<f32>;\n@group(0) @binding(2) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x;\n  if (i >= params.hidden_size) { return; }\n  let row_off = params.token_id * params.hidden_size;\n  output[i] = f32(embedding[row_off + i]) * params.embed_scale;\n}\n",
	fusedNormAdd: "enable f16;\nstruct Params { size: u32, eps: f32 }\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f16>;\n@group(0) @binding(2) var<storage, read> residual: array<f32>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n  let tid = lid.x;\n  let N = params.size;\n  var partial: f32 = 0.0;\n  var i = tid;\n  while (i < N) {\n    let val = input[i];\n    partial += val * val;\n    i += 256u;\n  }\n  shared_sum[tid] = partial;\n  workgroupBarrier();\n  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }\n    workgroupBarrier();\n  }\n  let rms = sqrt(shared_sum[0] / f32(N) + params.eps);\n  i = tid;\n  while (i < N) {\n    output[i] = residual[i] + input[i] * f32(weight[i]) / rms;\n    i += 256u;\n  }\n}\n",
	fusedPerHeadNormRope: "// Fuses per-head RMSNorm + RoPE into one kernel. Dispatched once per head\n// (workgroup_id.x = head index).\n//\n// RoPE base frequency can be attenuated via `rope_freqs`, a 256-element F16\n// table loaded at init. The source GGUF values are 1.0 (first 64 entries)\n// and 1e30 (entries 64..255, baking in `partial_rotary_factor=0.25`). In F16\n// storage these saturate to exactly 1.0 and +Inf — that's by design.\n// `base_freq / +Inf = 0` → `cos(0)=1, sin(0)=0` → no rotation, matching\n// Gemma 4's proportional-RoPE behavior on the trailing pairs of GLOBAL\n// (full-attention) layers. `apply_divisor=0` skips the lookup entirely for\n// sliding-window layers, which use the default RoPE schedule.\nenable f16;\nstruct Params { num_heads: u32, head_dim: u32, eps: f32, position: u32, theta: f32, apply_divisor: u32 }\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f16>;\n@group(0) @binding(2) var<uniform> params: Params;\n@group(0) @binding(3) var<storage, read> rope_freqs: array<f16>;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>,\n        @builtin(workgroup_id) wid: vec3<u32>) {\n  let head = wid.x;\n  if (head >= params.num_heads) { return; }\n  let tid = lid.x;\n  let base = head * params.head_dim;\n  var partial: f32 = 0.0;\n  var i = tid;\n  while (i < params.head_dim) {\n    let val = data[base + i];\n    partial += val * val;\n    i += 256u;\n  }\n  shared_sum[tid] = partial;\n  workgroupBarrier();\n  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }\n    workgroupBarrier();\n  }\n  let rms = sqrt(shared_sum[0] / f32(params.head_dim) + params.eps);\n  i = tid;\n  while (i < params.head_dim) {\n    data[base + i] = data[base + i] * f32(weight[i]) / rms;\n    i += 256u;\n  }\n  workgroupBarrier();\n  let half_dim = params.head_dim / 2u;\n  i = tid;\n  while (i < half_dim) {\n    let base_freq = 1.0 / pow(params.theta, f32(i * 2u) / f32(params.head_dim));\n    var freq: f32 = base_freq;\n    if (params.apply_divisor != 0u) {\n      freq = base_freq / f32(rope_freqs[i]);\n    }\n    let angle = f32(params.position) * freq;\n    let cos_a = cos(angle);\n    let sin_a = sin(angle);\n    let x0 = data[base + i];\n    let x1 = data[base + i + half_dim];\n    data[base + i] = x0 * cos_a - x1 * sin_a;\n    data[base + i + half_dim] = x0 * sin_a + x1 * cos_a;\n    i += 256u;\n  }\n}\n",
	geluMul: "const SQRT_2_OVER_PI: f32 = 0.7978845608028654;\nconst GELU_COEF_A: f32 = 0.044715;\n\n@group(0) @binding(0) var<storage, read> gate: array<f32>;\n@group(0) @binding(1) var<storage, read> up: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> size: u32;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x;\n  if (i >= size) { return; }\n  let x = gate[i];\n  let tanh_arg = clamp(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x), -15.0, 15.0);\n  let gelu = 0.5 * x * (1.0 + tanh(tanh_arg));\n  output[i] = gelu * up[i];\n}\n",
	kvCacheStore: "struct Params { num_kv_heads: u32, head_dim: u32, position: u32, max_seq_len: u32 }\n@group(0) @binding(0) var<storage, read> k_in: array<f32>;\n@group(0) @binding(1) var<storage, read> v_in: array<f32>;\n@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;\n@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x;\n  let total = params.num_kv_heads * params.head_dim;\n  if (i >= total) { return; }\n  let head = i / params.head_dim;\n  let d = i % params.head_dim;\n  let cache_idx = params.position * total + head * params.head_dim + d;\n  k_cache[cache_idx] = k_in[i];\n  v_cache[cache_idx] = v_in[i];\n}\n",
	logitSoftcap: "// Gemma 4 final-logit softcapping: `tanh(x / cap) * cap`, with the inner\n// argument clamped to ±15 to avoid `tanh` saturation breaking autograd-style\n// numerics (this is an inference-only engine, but the clamp is cheap and\n// matches the reference implementation).\nstruct Params { size: u32, cap: f32 }\n@group(0) @binding(0) var<storage, read_write> logits: array<f32>;\n@group(0) @binding(1) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x;\n  if (i >= params.size) { return; }\n  let x = logits[i] / params.cap;\n  let clamped = clamp(x, -15.0, 15.0);\n  logits[i] = tanh(clamped) * params.cap;\n}\n",
	matmulQuant: "// Single-output-row-per-workgroup matmul. All 256 threads cooperate on the\n// N-long dot product: thread `tid` reads weight[m, tid..N stride 256] and\n// input[tid..N stride 256] so consecutive threads within a warp hit\n// consecutive addresses (coalesced). Partial sums fold via an 8-level\n// shared-memory tree reduce.\n//\n// 2D dispatch support: callers with M > maxComputeWorkgroupsPerDimension\n// (lmHead's M=262144) use a sqrt-shaped 2D grid; `m = wg.y * ng.x + wg.x`\n// recovers the row index. 1D callers pass `ng.y = 1` so the formula\n// reduces to `m = wg.x`.\nenable f16;\nstruct Params { M: u32, N: u32 }\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f16>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\nconst WG: u32 = 256u;\nvar<workgroup> partials: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3<u32>,\n        @builtin(num_workgroups) ng: vec3<u32>,\n        @builtin(local_invocation_id) lid: vec3<u32>) {\n  let m = wg.y * ng.x + wg.x;\n  if (m >= params.M) { return; }\n  let tid = lid.x;\n  let N = params.N;\n  let row_off = m * N;\n\n  var acc: f32 = 0.0;\n  var k: u32 = tid;\n  loop {\n    if (k >= N) { break; }\n    acc = acc + f32(weight[row_off + k]) * input[k];\n    k = k + WG;\n  }\n  partials[tid] = acc;\n  workgroupBarrier();\n\n  var stride: u32 = 128u;\n  loop {\n    if (stride == 0u) { break; }\n    if (tid < stride) {\n      partials[tid] = partials[tid] + partials[tid + stride];\n    }\n    workgroupBarrier();\n    stride = stride >> 1u;\n  }\n\n  if (tid == 0u) {\n    output[m] = partials[0];\n  }\n}\n",
	matmulQuantMR4: "// Multi-row variant: each workgroup produces R=4 consecutive output rows,\n// amortizing tree-reduce + launch overhead across 4× more FMAs per\n// workgroup. Caller dispatches ceil(M/4) workgroups. Bindings and Params\n// layout match `matmul_quant` exactly, so bind-group layouts are\n// structurally identical (only the pipeline object differs). Used for\n// `ffn.linearGateUp` where M=I ∈ {6144, 12288} (both divisible by 4).\n//\n// Per-iteration k-loop: read `input[k]` once, multiply by 4 weights (one\n// per row), accumulate 4 partials. `input[k]` fetch amortizes 4×. Weight\n// fetches remain coalesced within each warp on the `tid` dimension; the\n// 4 per-row fetches are N f16s apart (= 3 KB for N=1536), so they issue\n// as separate memory transactions but each is independently coalesced.\n//\n// Reduce: all 4 rows share the 8-level tree-reduce — `partials` is 4×\n// wider (1024 f32 = 4 KB shared mem), and each stride step reduces 4\n// lanes in lockstep, keeping total barriers at 8.\nenable f16;\nstruct Params { M: u32, N: u32 }\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f16>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\nconst WG: u32 = 256u;\n// Pipeline-constant rows-per-workgroup. Injected at `createComputePipeline`\n// time from the active tuning profile's `matmul.rowsPerWorkgroupByKernel`\n// value for `ffn.linearGateUp`. The shader body is unrolled for R=4; other\n// values would require a variant shader (MR2, MR8) with matching unrolling\n// and a caller dispatch-count adjustment. See `src/tuning/profile.ts`.\noverride R: u32 = 4;\nvar<workgroup> partials: array<f32, 1024>; // 256 * 4\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3<u32>,\n        @builtin(num_workgroups) ng: vec3<u32>,\n        @builtin(local_invocation_id) lid: vec3<u32>) {\n  let m0 = (wg.y * ng.x + wg.x) * R;\n  let tid = lid.x;\n  let N = params.N;\n  let M = params.M;\n\n  var acc0: f32 = 0.0;\n  var acc1: f32 = 0.0;\n  var acc2: f32 = 0.0;\n  var acc3: f32 = 0.0;\n\n  // Clamp each row to a valid index so out-of-range rows (when M % 4 != 0, e.g.\n  // some attention projections) read in-bounds; their output is never written\n  // below. No-op for the FFN dims this was written for (M divisible by 4).\n  let row0 = min(m0, M - 1u) * N;\n  let row1 = min(m0 + 1u, M - 1u) * N;\n  let row2 = min(m0 + 2u, M - 1u) * N;\n  let row3 = min(m0 + 3u, M - 1u) * N;\n\n  var k: u32 = tid;\n  loop {\n    if (k >= N) { break; }\n    let inp = input[k];\n    acc0 = acc0 + f32(weight[row0 + k]) * inp;\n    acc1 = acc1 + f32(weight[row1 + k]) * inp;\n    acc2 = acc2 + f32(weight[row2 + k]) * inp;\n    acc3 = acc3 + f32(weight[row3 + k]) * inp;\n    k = k + WG;\n  }\n\n  partials[0u * WG + tid] = acc0;\n  partials[1u * WG + tid] = acc1;\n  partials[2u * WG + tid] = acc2;\n  partials[3u * WG + tid] = acc3;\n  workgroupBarrier();\n\n  var stride: u32 = 128u;\n  loop {\n    if (stride == 0u) { break; }\n    if (tid < stride) {\n      partials[0u * WG + tid] = partials[0u * WG + tid] + partials[0u * WG + tid + stride];\n      partials[1u * WG + tid] = partials[1u * WG + tid] + partials[1u * WG + tid + stride];\n      partials[2u * WG + tid] = partials[2u * WG + tid] + partials[2u * WG + tid + stride];\n      partials[3u * WG + tid] = partials[3u * WG + tid] + partials[3u * WG + tid + stride];\n    }\n    workgroupBarrier();\n    stride = stride >> 1u;\n  }\n\n  if (tid < R) {\n    let m = m0 + tid;\n    if (m < M) {\n      output[m] = partials[tid * WG];\n    }\n  }\n}\n",
	matmulQ8: "// In-shader Q8_0 matmul (GEMV): output[m] = Σ_k dequant(qweight[m,k]) · input[k].\n//\n// Weights are kept QUANTIZED in GPU memory (1 byte/weight vs F16's 2) and\n// dequantized inside the dot-product loop — the memory lever that lets larger\n// models fit. Symmetric int8 with one f16 scale per 32-element block (GGUF\n// Q8_0 layout, re-derived from the source weights at load):\n//   qweight : int8, row-major [M, N], packed 4-per-u32 over the flattened array\n//   qscale  : f16,  [M, N/32]  (one scale per 32-wide block per row)\n// dequant(w[m,k]) = f32(int8) · scale[m, k/32].\n//\n// Bandwidth-bound GEMV: halving weight bytes read should also help tps, the\n// same reason ORT's MatMulNBits / the LFM2 engine keep weights quantized.\nenable f16;\nstruct Params { M: u32, N: u32 }\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> qweight: array<u32>;\n@group(0) @binding(2) var<storage, read> qscale: array<f16>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\nconst WG: u32 = 256u;\nvar<workgroup> partials: array<f32, 256>;\n\n// Extract the `lane`-th signed int8 (little-endian) from a packed u32.\nfn unpack_i8(packed: u32, lane: u32) -> i32 {\n  let b = (packed >> (lane * 8u)) & 0xffu;\n  return select(i32(b), i32(b) - 256, b >= 128u);\n}\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3<u32>,\n        @builtin(num_workgroups) ng: vec3<u32>,\n        @builtin(local_invocation_id) lid: vec3<u32>) {\n  let m = wg.y * ng.x + wg.x;\n  if (m >= params.M) { return; }\n  let tid = lid.x;\n  let N = params.N;\n  let nblocks = N / 32u;\n  let row_q_base = m * N;        // flattened int8 element index for row m\n  let row_s_base = m * nblocks;  // scale index for row m\n\n  var acc: f32 = 0.0;\n  var k: u32 = tid;\n  loop {\n    if (k >= N) { break; }\n    let gi = row_q_base + k;\n    let qv = unpack_i8(qweight[gi >> 2u], gi & 3u);\n    let scale = f32(qscale[row_s_base + (k >> 5u)]);\n    acc = acc + f32(qv) * scale * input[k];\n    k = k + WG;\n  }\n  partials[tid] = acc;\n  workgroupBarrier();\n\n  var stride: u32 = 128u;\n  loop {\n    if (stride == 0u) { break; }\n    if (tid < stride) {\n      partials[tid] = partials[tid] + partials[tid + stride];\n    }\n    workgroupBarrier();\n    stride = stride >> 1u;\n  }\n\n  if (tid == 0u) {\n    output[m] = partials[0];\n  }\n}\n",
	matmulQ4K: "// In-shader 4-bit matmul (GEMV): output[m] = Σ_k dequant(qweight[m,k]) · input[k].\n//\n// The 4-bit memory lever — the largest one. Weights are kept at 4 bits/value in\n// GPU memory (vs F16's 16, Q8's 8) and dequantized inside the dot-product loop,\n// the same idea as ORT's MatMulNBits. This is what lets the 4–8B class fit.\n//\n// Block-affine 4-bit, mirroring GGUF Q4_K's *structure* (256-element super-block\n// split into 8 sub-blocks of 32, each sub-block an independent 4-bit affine grid)\n// but keeping each sub-block's scale+min as f16 rather than packing them to 6 bits\n// via a super-scale. Trading ~0.5 bit/weight of packing for a cleaner, strictly\n// higher-precision representation than GGUF Q4_K — so a Q4_K *source* tensor round-\n// trips near-losslessly, and only genuinely-higher-precision source tensors (Q6_K)\n// take a real down-quant:\n//   qweight : 4-bit unsigned, row-major [M, N], packed 8-per-u32 over the flat array\n//   qmeta    : f16, [M, N/256, 16] — per super-block, 8 scales then 8 mins\n// dequant(w[m,k]) = scale[m, sb, sub] · f32(nibble) + min[m, sb, sub],\n//   where sb = (k mod N)/256, sub = (k mod 256)/32, nibble ∈ [0,15].\nenable f16;\nstruct Params { M: u32, N: u32 }\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> qweight: array<u32>;\n@group(0) @binding(2) var<storage, read> qmeta: array<f16>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\nconst WG: u32 = 256u;\nvar<workgroup> partials: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3<u32>,\n        @builtin(num_workgroups) ng: vec3<u32>,\n        @builtin(local_invocation_id) lid: vec3<u32>) {\n  let m = wg.y * ng.x + wg.x;\n  if (m >= params.M) { return; }\n  let tid = lid.x;\n  let N = params.N;\n  let nsblk = N / 256u;             // super-blocks per row\n  let row_q_base = m * N;           // flat 4-bit element index for row m\n  let row_qmeta_base = m * nsblk * 16u;  // f16 qmeta index for row m\n\n  var acc: f32 = 0.0;\n  var k: u32 = tid;\n  loop {\n    if (k >= N) { break; }\n    // Unpack the 4-bit weight: 8 nibbles per u32, little-endian nibble order.\n    let fi = row_q_base + k;\n    let nib = (qweight[fi >> 3u] >> ((fi & 7u) * 4u)) & 0xfu;\n    // Locate the sub-block this element belongs to and read its affine pair.\n    let sb = k >> 8u;              // k / 256\n    let sub = (k & 255u) >> 5u;    // (k % 256) / 32  ∈ [0,8)\n    let mbase = row_qmeta_base + sb * 16u;\n    let scale = f32(qmeta[mbase + sub]);\n    let mn = f32(qmeta[mbase + 8u + sub]);\n    acc = acc + (scale * f32(nib) + mn) * input[k];\n    k = k + WG;\n  }\n  partials[tid] = acc;\n  workgroupBarrier();\n\n  var stride: u32 = 128u;\n  loop {\n    if (stride == 0u) { break; }\n    if (tid < stride) {\n      partials[tid] = partials[tid] + partials[tid + stride];\n    }\n    workgroupBarrier();\n    stride = stride >> 1u;\n  }\n\n  if (tid == 0u) {\n    output[m] = partials[0];\n  }\n}\n",
	matmulQ4KMr: "// Multi-row variant of matmul_q4k (R=4 output rows per workgroup), the speed\n// half of the 4-bit lever. Mirrors matmul_quant_mr4's structure — read input[k]\n// once, multiply into R per-row accumulators, share the 8-level tree-reduce —\n// but each row dequantizes its own 4-bit block-affine weight in the loop.\n// Caller dispatches ceil(M/R) workgroups. Bindings/Params match matmul_q4k\n// exactly, so the bind-group layout is structurally identical (only the pipeline\n// object differs). The block-affine sub-position (sb, sub) is the same for all R\n// rows at a given k, so it's computed once per iteration.\n//\n// Why this is the win at batch-1 decode: the GEMV is launch-overhead/occupancy-\n// bound, not bandwidth-bound (~9 GB/s effective vs ~200+ GB/s peak), so 4× fewer\n// workgroups + amortized input/reduce buys more than any memory-traffic tweak.\nenable f16;\nstruct Params { M: u32, N: u32 }\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> qweight: array<u32>;\n@group(0) @binding(2) var<storage, read> qmeta: array<f16>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\nconst WG: u32 = 256u;\nconst R: u32 = 4u;\nvar<workgroup> partials: array<f32, 1024>; // 256 * 4\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3<u32>,\n        @builtin(num_workgroups) ng: vec3<u32>,\n        @builtin(local_invocation_id) lid: vec3<u32>) {\n  let m0 = (wg.y * ng.x + wg.x) * R;\n  let tid = lid.x;\n  let N = params.N;\n  let M = params.M;\n  let nsblk = N / 256u;\n\n  // Clamp each row to a valid index so out-of-range rows (when M % 4 != 0) read\n  // in-bounds data; their output is simply never written below.\n  let mc0 = min(m0, M - 1u);\n  let mc1 = min(m0 + 1u, M - 1u);\n  let mc2 = min(m0 + 2u, M - 1u);\n  let mc3 = min(m0 + 3u, M - 1u);\n  let q0 = mc0 * N;       let s0 = mc0 * nsblk * 16u;\n  let q1 = mc1 * N;       let s1 = mc1 * nsblk * 16u;\n  let q2 = mc2 * N;       let s2 = mc2 * nsblk * 16u;\n  let q3 = mc3 * N;       let s3 = mc3 * nsblk * 16u;\n\n  var acc0: f32 = 0.0;\n  var acc1: f32 = 0.0;\n  var acc2: f32 = 0.0;\n  var acc3: f32 = 0.0;\n\n  var k: u32 = tid;\n  loop {\n    if (k >= N) { break; }\n    let inp = input[k];\n    let sb = k >> 8u;\n    let sub = (k & 255u) >> 5u;\n    let moff = sb * 16u + sub;       // scale offset within a row's meta\n    let shift = (k & 7u) * 4u;\n    let widx = k >> 3u;              // u32-word offset within a row's quants\n\n    let nib0 = (qweight[(q0 >> 3u) + widx] >> shift) & 0xfu;\n    acc0 = acc0 + (f32(qmeta[s0 + moff]) * f32(nib0) + f32(qmeta[s0 + moff + 8u])) * inp;\n    let nib1 = (qweight[(q1 >> 3u) + widx] >> shift) & 0xfu;\n    acc1 = acc1 + (f32(qmeta[s1 + moff]) * f32(nib1) + f32(qmeta[s1 + moff + 8u])) * inp;\n    let nib2 = (qweight[(q2 >> 3u) + widx] >> shift) & 0xfu;\n    acc2 = acc2 + (f32(qmeta[s2 + moff]) * f32(nib2) + f32(qmeta[s2 + moff + 8u])) * inp;\n    let nib3 = (qweight[(q3 >> 3u) + widx] >> shift) & 0xfu;\n    acc3 = acc3 + (f32(qmeta[s3 + moff]) * f32(nib3) + f32(qmeta[s3 + moff + 8u])) * inp;\n    k = k + WG;\n  }\n\n  partials[0u * WG + tid] = acc0;\n  partials[1u * WG + tid] = acc1;\n  partials[2u * WG + tid] = acc2;\n  partials[3u * WG + tid] = acc3;\n  workgroupBarrier();\n\n  var stride: u32 = 128u;\n  loop {\n    if (stride == 0u) { break; }\n    if (tid < stride) {\n      partials[0u * WG + tid] = partials[0u * WG + tid] + partials[0u * WG + tid + stride];\n      partials[1u * WG + tid] = partials[1u * WG + tid] + partials[1u * WG + tid + stride];\n      partials[2u * WG + tid] = partials[2u * WG + tid] + partials[2u * WG + tid + stride];\n      partials[3u * WG + tid] = partials[3u * WG + tid] + partials[3u * WG + tid + stride];\n    }\n    workgroupBarrier();\n    stride = stride >> 1u;\n  }\n\n  if (tid < R) {\n    let m = m0 + tid;\n    if (m < M) {\n      output[m] = partials[tid * WG];\n    }\n  }\n}\n",
	matmulQ8Mr: "// Multi-row variant of matmul_q8 (R=4 output rows per workgroup) — the Q8 speed\n// path, mirroring matmul_q4k_mr. Read input[k] once, dequantize 4 rows' int8\n// weights in the loop, share the 8-level tree-reduce. Caller dispatches\n// ceil(M/R) workgroups; bindings/Params match matmul_q8 exactly so the\n// bind-group layout is structurally identical (only the pipeline differs).\n// N % 32 == 0 (Q8 block) ⇒ each row base m·N is a multiple of 4, so the packed\n// int8 word index decomposes as (m·N>>2)+(k>>2) with lane k&3.\nenable f16;\nstruct Params { M: u32, N: u32 }\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> qweight: array<u32>;\n@group(0) @binding(2) var<storage, read> qscale: array<f16>;\n@group(0) @binding(3) var<storage, read_write> output: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\n\nconst WG: u32 = 256u;\nconst R: u32 = 4u;\nvar<workgroup> partials: array<f32, 1024>; // 256 * 4\n\nfn unpack_i8(packed: u32, lane: u32) -> i32 {\n  let b = (packed >> (lane * 8u)) & 0xffu;\n  return select(i32(b), i32(b) - 256, b >= 128u);\n}\n\n@compute @workgroup_size(256)\nfn main(@builtin(workgroup_id) wg: vec3<u32>,\n        @builtin(num_workgroups) ng: vec3<u32>,\n        @builtin(local_invocation_id) lid: vec3<u32>) {\n  let m0 = (wg.y * ng.x + wg.x) * R;\n  let tid = lid.x;\n  let N = params.N;\n  let M = params.M;\n  let nblocks = N / 32u;\n\n  // Clamp each row to a valid index so out-of-range rows (M % 4 != 0) read\n  // in-bounds; their output is never written below.\n  let mc0 = min(m0, M - 1u);\n  let mc1 = min(m0 + 1u, M - 1u);\n  let mc2 = min(m0 + 2u, M - 1u);\n  let mc3 = min(m0 + 3u, M - 1u);\n  let q0 = mc0 * N;  let s0 = mc0 * nblocks;\n  let q1 = mc1 * N;  let s1 = mc1 * nblocks;\n  let q2 = mc2 * N;  let s2 = mc2 * nblocks;\n  let q3 = mc3 * N;  let s3 = mc3 * nblocks;\n\n  var acc0: f32 = 0.0;\n  var acc1: f32 = 0.0;\n  var acc2: f32 = 0.0;\n  var acc3: f32 = 0.0;\n\n  var k: u32 = tid;\n  loop {\n    if (k >= N) { break; }\n    let inp = input[k];\n    let widx = k >> 2u;\n    let lane = k & 3u;\n    let sblk = k >> 5u;\n    acc0 = acc0 + f32(unpack_i8(qweight[(q0 >> 2u) + widx], lane)) * f32(qscale[s0 + sblk]) * inp;\n    acc1 = acc1 + f32(unpack_i8(qweight[(q1 >> 2u) + widx], lane)) * f32(qscale[s1 + sblk]) * inp;\n    acc2 = acc2 + f32(unpack_i8(qweight[(q2 >> 2u) + widx], lane)) * f32(qscale[s2 + sblk]) * inp;\n    acc3 = acc3 + f32(unpack_i8(qweight[(q3 >> 2u) + widx], lane)) * f32(qscale[s3 + sblk]) * inp;\n    k = k + WG;\n  }\n\n  partials[0u * WG + tid] = acc0;\n  partials[1u * WG + tid] = acc1;\n  partials[2u * WG + tid] = acc2;\n  partials[3u * WG + tid] = acc3;\n  workgroupBarrier();\n\n  var stride: u32 = 128u;\n  loop {\n    if (stride == 0u) { break; }\n    if (tid < stride) {\n      partials[0u * WG + tid] = partials[0u * WG + tid] + partials[0u * WG + tid + stride];\n      partials[1u * WG + tid] = partials[1u * WG + tid] + partials[1u * WG + tid + stride];\n      partials[2u * WG + tid] = partials[2u * WG + tid] + partials[2u * WG + tid + stride];\n      partials[3u * WG + tid] = partials[3u * WG + tid] + partials[3u * WG + tid + stride];\n    }\n    workgroupBarrier();\n    stride = stride >> 1u;\n  }\n\n  if (tid < R) {\n    let m = m0 + tid;\n    if (m < M) { output[m] = partials[tid * WG]; }\n  }\n}\n",
	perHeadRmsNorm: "enable f16;\nstruct Params { num_heads: u32, head_dim: u32, eps: f32, pad: u32 }\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f16>;\n@group(0) @binding(2) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n  let tid = lid.x;\n  let head = wid.x;\n  if (head >= params.num_heads) { return; }\n  let base = head * params.head_dim;\n  var partial: f32 = 0.0;\n  var i = tid;\n  while (i < params.head_dim) {\n    let val = data[base + i];\n    partial += val * val;\n    i += 256u;\n  }\n  shared_sum[tid] = partial;\n  workgroupBarrier();\n  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }\n    workgroupBarrier();\n  }\n  let rms = sqrt(shared_sum[0] / f32(params.head_dim) + params.eps);\n  i = tid;\n  while (i < params.head_dim) {\n    data[base + i] = data[base + i] * f32(weight[i]) / rms;\n    i += 256u;\n  }\n}\n",
	perHeadRmsNormNoWeight: "// Per-head RMSNorm WITHOUT a learned weight — equivalent to `x / rms(x)` per head.\n//\n// Gemma 4's v_norm uses this form: HF's `Gemma4TextAttention.v_norm` applies RMSNorm\n// with weight ≡ 1, and the GGUF export drops `attn_v_norm` as trivial, so we have no\n// per-tensor weight to multiply by. Must run on V between linearV and kvCacheStore so\n// the cached (and subsequently shared-to-consumer-layer) V is already normalized.\nenable f16;\nstruct Params { num_heads: u32, head_dim: u32, eps: f32, pad: u32 }\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n  let tid = lid.x;\n  let head = wid.x;\n  if (head >= params.num_heads) { return; }\n  let base = head * params.head_dim;\n  var partial: f32 = 0.0;\n  var i = tid;\n  while (i < params.head_dim) {\n    let val = data[base + i];\n    partial += val * val;\n    i += 256u;\n  }\n  shared_sum[tid] = partial;\n  workgroupBarrier();\n  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }\n    workgroupBarrier();\n  }\n  let rms = sqrt(shared_sum[0] / f32(params.head_dim) + params.eps);\n  i = tid;\n  while (i < params.head_dim) {\n    data[base + i] = data[base + i] / rms;\n    i += 256u;\n  }\n}\n",
	pleGeluMul: "// Per-layer embedding (PLE) Stage 2 step b: GELU the `inp_gate` result and\n// multiply elementwise by the per-layer slice of `ple_inputs`. Dispatched\n// once per layer (inside the block).\nconst SQRT_2_OVER_PI: f32 = 0.7978845608028654;\nconst GELU_COEF_A: f32 = 0.044715;\nstruct Params { layer_offset: u32, size: u32 }\n@group(0) @binding(0) var<storage, read> gate: array<f32>;\n@group(0) @binding(1) var<storage, read> ple_inputs: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x;\n  if (i >= params.size) { return; }\n  let x = gate[i];\n  let tanh_arg = clamp(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x), -15.0, 15.0);\n  let gelu = 0.5 * x * (1.0 + tanh(tanh_arg));\n  output[i] = gelu * ple_inputs[params.layer_offset + i];\n}\n",
	pleSkipScaleAdd: "// Per-layer embedding (PLE) block skip-scale:\n//   `hidden = (hidden + ple_residual) * layer_output_scale`\n//\n// `layer_output_scale` is a per-layer F16 scalar (stored as a [1]-element\n// buffer, padded to 4 bytes). Dispatched once per layer at end of block.\nenable f16;\nstruct Params { size: u32 }\n@group(0) @binding(0) var<storage, read_write> hidden: array<f32>;\n@group(0) @binding(1) var<storage, read> post_normed: array<f32>;\n@group(0) @binding(2) var<storage, read> scale: array<f16>;\n@group(0) @binding(3) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x;\n  if (i >= params.size) { return; }\n  let s = f32(scale[0]);\n  hidden[i] = (hidden[i] + post_normed[i]) * s;\n}\n",
	pleStage1Fuse: "// Per-layer embedding (PLE) Stage 1. Fuses per-layer-projected norm +\n// per-layer embedding lookup + rsqrt(2) blend into a single kernel.\n//\n// Dispatched once per layer per forward pass. Each call binds its layer's\n// per_layer_token_embd slice (~128 MB on Gemma 4 E2B). The `projected`\n// input comes from `per_layer_model_proj @ hidden`, reshaped as\n// `[num_layers, per_layer_dim]`.\nenable f16;\nstruct Params { layer_idx: u32, token_id: u32, per_layer_dim: u32, eps: f32 }\n@group(0) @binding(0) var<storage, read> projected: array<f32>;\n@group(0) @binding(1) var<storage, read> norm_weight: array<f16>;\n@group(0) @binding(2) var<storage, read> embed_slice: array<f16>;\n@group(0) @binding(3) var<storage, read_write> ple_inputs: array<f32>;\n@group(0) @binding(4) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n  let tid = lid.x;\n  let D = params.per_layer_dim;\n  let proj_base = params.layer_idx * D;\n  var partial: f32 = 0.0;\n  if (tid < D) {\n    let v = projected[proj_base + tid];\n    partial = v * v;\n  }\n  shared_sum[tid] = partial;\n  workgroupBarrier();\n  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }\n    workgroupBarrier();\n  }\n  let rms = sqrt(shared_sum[0] / f32(D) + params.eps);\n  if (tid < D) {\n    let proj_normed = (projected[proj_base + tid] / rms) * f32(norm_weight[tid]);\n    let embed_val = f32(embed_slice[params.token_id * D + tid]) * sqrt(f32(D));\n    ple_inputs[proj_base + tid] = (proj_normed + embed_val) * inverseSqrt(2.0);\n  }\n}\n",
	rmsNorm: "enable f16;\nstruct Params { hidden_size: u32, eps: f32 }\n@group(0) @binding(0) var<storage, read> input: array<f32>;\n@group(0) @binding(1) var<storage, read> weight: array<f16>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> params: Params;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n  let tid = lid.x;\n  let hidden_size = params.hidden_size;\n  var partial_sum: f32 = 0.0;\n  var i = tid;\n  while (i < hidden_size) {\n    let val = input[i];\n    partial_sum += val * val;\n    i += 256u;\n  }\n  shared_sum[tid] = partial_sum;\n  workgroupBarrier();\n  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }\n    workgroupBarrier();\n  }\n  let rms = sqrt(shared_sum[0] / f32(hidden_size) + params.eps);\n  i = tid;\n  while (i < hidden_size) {\n    output[i] = input[i] * f32(weight[i]) / rms;\n    i += 256u;\n  }\n}\n",
	rope: "struct Params { num_heads: u32, head_dim: u32, position: u32, theta: f32 }\n@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n@group(0) @binding(1) var<uniform> params: Params;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let idx = gid.x;\n  let half_dim = params.head_dim / 2u;\n  let total_pairs = params.num_heads * half_dim;\n  if (idx >= total_pairs) { return; }\n  let head = idx / half_dim;\n  let i = idx % half_dim;\n  let base = head * params.head_dim;\n  let freq = 1.0 / pow(params.theta, f32(i * 2u) / f32(params.head_dim));\n  let angle = f32(params.position) * freq;\n  let cos_a = cos(angle);\n  let sin_a = sin(angle);\n  let x0 = data[base + i];\n  let x1 = data[base + i + half_dim];\n  data[base + i] = x0 * cos_a - x1 * sin_a;\n  data[base + i + half_dim] = x0 * sin_a + x1 * cos_a;\n}\n",
	siluMul: "// SwiGLU activation: out = SiLU(gate) * up, where SiLU(x) = x * sigmoid(x).\n// Mirrors gelu_mul.wgsl (same binding layout / dispatch) so the FFN path can\n// switch GELU↔SiLU on a config flag with no other plumbing change. Qwen3 (and\n// the llama/Qwen MLP family) use SiLU here where Gemma 4 uses tanh-GELU.\n@group(0) @binding(0) var<storage, read> gate: array<f32>;\n@group(0) @binding(1) var<storage, read> up: array<f32>;\n@group(0) @binding(2) var<storage, read_write> output: array<f32>;\n@group(0) @binding(3) var<uniform> size: u32;\n\n@compute @workgroup_size(256)\nfn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let i = gid.x;\n  if (i >= size) { return; }\n  let x = gate[i];\n  // SiLU(x) = x * sigmoid(x) = x / (1 + e^-x). Clamp the exponent argument to\n  // keep parity with gelu_mul.wgsl's overflow-guard discipline on f32.\n  let silu = x / (1.0 + exp(-clamp(x, -88.0, 88.0)));\n  output[i] = silu * up[i];\n}\n",
	softmax: "struct Params { num_heads: u32, seq_len: u32 }\n@group(0) @binding(0) var<storage, read_write> scores: array<f32>;\n@group(0) @binding(1) var<uniform> params: Params;\nvar<workgroup> shared_max: array<f32, 256>;\nvar<workgroup> shared_sum: array<f32, 256>;\n\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {\n  let tid = lid.x;\n  let head = wid.x;\n  if (head >= params.num_heads) { return; }\n  let base = head * params.seq_len;\n  var local_max: f32 = -1e30;\n  var i = tid;\n  while (i < params.seq_len) {\n    local_max = max(local_max, scores[base + i]);\n    i += 256u;\n  }\n  shared_max[tid] = local_max;\n  workgroupBarrier();\n  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) { shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]); }\n    workgroupBarrier();\n  }\n  let max_val = shared_max[0];\n  var local_sum: f32 = 0.0;\n  i = tid;\n  while (i < params.seq_len) {\n    let e = exp(scores[base + i] - max_val);\n    scores[base + i] = e;\n    local_sum += e;\n    i += 256u;\n  }\n  shared_sum[tid] = local_sum;\n  workgroupBarrier();\n  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {\n    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }\n    workgroupBarrier();\n  }\n  let sum_val = shared_sum[0];\n  i = tid;\n  while (i < params.seq_len) {\n    scores[base + i] = scores[base + i] / sum_val;\n    i += 256u;\n  }\n}\n",
	topk256: "@group(0) @binding(0) var<storage, read> logits: array<f32>;\n@group(0) @binding(1) var<storage, read_write> result: array<f32>;\n@group(0) @binding(2) var<uniform> size: u32;\n\n@compute @workgroup_size(256)\nfn main(@builtin(local_invocation_id) lid: vec3<u32>) {\n  let tid = lid.x;\n  var local_max: f32 = -1e30;\n  var local_idx: u32 = 0u;\n  var i = tid;\n  while (i < size) {\n    let val = logits[i];\n    if (val > local_max) {\n      local_max = val;\n      local_idx = i;\n    }\n    i += 256u;\n  }\n  result[tid * 2u] = local_max;\n  result[tid * 2u + 1u] = bitcast<f32>(local_idx);\n}\n"
}, t = class {
	buffer;
	view;
	offset;
	textDecoder;
	constructor(e) {
		e instanceof Uint8Array ? (this.buffer = e.buffer, this.view = new DataView(this.buffer, e.byteOffset, e.byteLength), this.offset = 0) : (this.buffer = e, this.view = new DataView(this.buffer), this.offset = 0), this.textDecoder = new TextDecoder("utf-8");
	}
	readUint32() {
		let e = this.view.getUint32(this.offset, !0);
		return this.offset += 4, e;
	}
	readUint64() {
		let e = this.view.getBigUint64(this.offset, !0);
		return this.offset += 8, e;
	}
	readString() {
		let e = Number(this.readUint64()), t = new Uint8Array(this.buffer, this.offset, e);
		return this.offset += e, this.textDecoder.decode(t);
	}
	readValue(e) {
		switch (e) {
			case 0: {
				let e = this.view.getUint8(this.offset);
				return this.offset += 1, {
					type: "uint8",
					value: e
				};
			}
			case 1: {
				let e = this.view.getInt8(this.offset);
				return this.offset += 1, {
					type: "int8",
					value: e
				};
			}
			case 2: {
				let e = this.view.getUint16(this.offset, !0);
				return this.offset += 2, {
					type: "uint16",
					value: e
				};
			}
			case 3: {
				let e = this.view.getInt16(this.offset, !0);
				return this.offset += 2, {
					type: "int16",
					value: e
				};
			}
			case 4: return {
				type: "uint32",
				value: this.readUint32()
			};
			case 5: {
				let e = this.view.getInt32(this.offset, !0);
				return this.offset += 4, {
					type: "int32",
					value: e
				};
			}
			case 6: {
				let e = this.view.getFloat32(this.offset, !0);
				return this.offset += 4, {
					type: "float32",
					value: e
				};
			}
			case 7: {
				let e = this.view.getUint8(this.offset);
				return this.offset += 1, {
					type: "bool",
					value: e !== 0
				};
			}
			case 8: return {
				type: "string",
				value: this.readString()
			};
			case 9: {
				let e = this.readUint32(), t = Number(this.readUint64()), n = [];
				for (let r = 0; r < t; r++) n.push(this.readValue(e));
				return {
					type: "array",
					value: n
				};
			}
			case 10: return {
				type: "uint64",
				value: this.readUint64()
			};
			case 11: {
				let e = this.view.getBigInt64(this.offset, !0);
				return this.offset += 8, {
					type: "int64",
					value: e
				};
			}
			case 12: {
				let e = this.view.getFloat64(this.offset, !0);
				return this.offset += 8, {
					type: "float64",
					value: e
				};
			}
			default: return {
				type: "unknown",
				value: null
			};
		}
	}
	parse() {
		let e = this.readUint32();
		if (e !== 1179993927) throw Error(`Invalid GGUF magic: 0x${e.toString(16)}`);
		let t = this.readUint32(), n = this.readUint64(), r = this.readUint64(), i = /* @__PURE__ */ new Map();
		for (let e = 0n; e < r; e++) {
			let e = this.readString(), t = this.readUint32(), n = this.readValue(t);
			i.set(e, n);
		}
		let a = [];
		for (let e = 0n; e < n; e++) {
			let e = this.readString(), t = this.readUint32(), n = [];
			for (let e = 0; e < t; e++) n.push(this.readUint64());
			let r = this.readUint32(), i = this.readUint64();
			a.push({
				name: e,
				dims: n,
				type: r,
				offset: i
			});
		}
		return {
			version: t,
			tensor_count: n,
			kv_count: r,
			kv: i,
			tensors: a,
			dataOffset: Math.ceil(this.offset / 32) * 32
		};
	}
	f16ToF32(e) {
		let t = e >> 15 & 1, n = e >> 10 & 31, r = e & 1023;
		return n === 0 ? r === 0 ? t ? -0 : 0 : (t ? -1 : 1) * 2 ** -14 * (r / 1024) : n === 31 ? r === 0 ? t ? -Infinity : Infinity : NaN : (t ? -1 : 1) * 2 ** (n - 15) * (1 + r / 1024);
	}
	dequantizeQ8_0(e, t) {
		let n = new Float32Array(t), r = 0, i = e;
		for (; r < t;) {
			let e = this.view.getUint16(i, !0), a = this.f16ToF32(e);
			i += 2;
			let o = Math.min(32, t - r);
			for (let e = 0; e < o; e++) {
				let t = this.view.getInt8(i + e);
				n[r++] = t * a;
			}
			i += 32;
		}
		return n;
	}
	dequantizeF16(e, t) {
		let n = new Float32Array(t);
		for (let r = 0; r < t; r++) {
			let t = this.view.getUint16(e + r * 2, !0);
			n[r] = this.f16ToF32(t);
		}
		return n;
	}
	dequantizeBF16(e, t) {
		let n = new Float32Array(t), r = /* @__PURE__ */ new ArrayBuffer(4), i = new Uint32Array(r), a = new Float32Array(r);
		for (let r = 0; r < t; r++) i[0] = this.view.getUint16(e + r * 2, !0) << 16, n[r] = a[0];
		return n;
	}
	dequantizeQ4_K(e, t) {
		let n = new Float32Array(t), r = 0, i = e, a = new Uint8Array(12), o = (e) => e < 4 ? [a[e] & 63, a[e + 4] & 63] : [a[e + 4] & 15 | a[e - 4] >> 6 << 4, a[e + 4] >> 4 | a[e] >> 6 << 4];
		for (; r < t;) {
			let e = this.f16ToF32(this.view.getUint16(i, !0)), s = this.f16ToF32(this.view.getUint16(i + 2, !0));
			for (let e = 0; e < 12; e++) a[e] = this.view.getUint8(i + 4 + e);
			let c = i + 16;
			for (let i = 0; i < 4 && r < t; i++) {
				let [a, l] = o(i * 2), [u, d] = o(i * 2 + 1), f = e * a, p = s * l, m = e * u, h = s * d;
				for (let e = 0; e < 32 && r < t; e++) {
					let t = this.view.getUint8(c + i * 32 + e);
					n[r++] = f * (t & 15) - p;
				}
				for (let e = 0; e < 32 && r < t; e++) {
					let t = this.view.getUint8(c + i * 32 + e);
					n[r++] = m * (t >> 4 & 15) - h;
				}
			}
			i += 144;
		}
		return n;
	}
	decodeQ5_KBlock(e, t, n) {
		let r = new Uint8Array(12), i = (e) => e < 4 ? [r[e] & 63, r[e + 4] & 63] : [r[e + 4] & 15 | r[e - 4] >> 6 << 4, r[e + 4] >> 4 | r[e] >> 6 << 4], a = this.f16ToF32(this.view.getUint16(e, !0)), o = this.f16ToF32(this.view.getUint16(e + 2, !0));
		for (let t = 0; t < 12; t++) r[t] = this.view.getUint8(e + 4 + t);
		let s = e + 16, c = e + 48, l = 1, u = 2, d = n;
		for (let e = 0; e < 4; e++) {
			let [n, r] = i(e * 2), [f, p] = i(e * 2 + 1), m = a * n, h = o * r, g = a * f, _ = o * p;
			for (let n = 0; n < 32; n++) {
				let r = this.view.getUint8(c + e * 32 + n), i = this.view.getUint8(s + n), a = r & 15, o = i & l ? 16 : 0;
				t[d++] = m * (a + o) - h;
			}
			for (let n = 0; n < 32; n++) {
				let r = this.view.getUint8(c + e * 32 + n), i = this.view.getUint8(s + n), a = r >> 4 & 15, o = i & u ? 16 : 0;
				t[d++] = g * (a + o) - _;
			}
			l <<= 2, u <<= 2;
		}
	}
	dequantizeQ5_K(e, t) {
		let n = new Float32Array(t), r = 0, i = e, a = new Uint8Array(12), o = (e) => e < 4 ? [a[e] & 63, a[e + 4] & 63] : [a[e + 4] & 15 | a[e - 4] >> 6 << 4, a[e + 4] >> 4 | a[e] >> 6 << 4];
		for (; r < t;) {
			let e = this.f16ToF32(this.view.getUint16(i, !0)), s = this.f16ToF32(this.view.getUint16(i + 2, !0));
			for (let e = 0; e < 12; e++) a[e] = this.view.getUint8(i + 4 + e);
			let c = i + 16, l = i + 48, u = 1, d = 2;
			for (let i = 0; i < 4 && r < t; i++) {
				let [a, f] = o(i * 2), [p, m] = o(i * 2 + 1), h = e * a, g = s * f, _ = e * p, v = s * m;
				for (let e = 0; e < 32 && r < t; e++) {
					let t = this.view.getUint8(l + i * 32 + e), a = this.view.getUint8(c + e), o = t & 15, s = a & u ? 16 : 0;
					n[r++] = h * (o + s) - g;
				}
				for (let e = 0; e < 32 && r < t; e++) {
					let t = this.view.getUint8(l + i * 32 + e), a = this.view.getUint8(c + e), o = t >> 4 & 15, s = a & d ? 16 : 0;
					n[r++] = _ * (o + s) - v;
				}
				u <<= 2, d <<= 2;
			}
			i += 176;
		}
		return n;
	}
	dequantizeQ6_K(e, t) {
		let n = new Float32Array(t), r = 0, i = e;
		for (; r < t;) {
			let e = i, a = i + 128, o = i + 192, s = this.f16ToF32(this.view.getUint16(i + 208, !0));
			for (let i = 0; i < 2 && r < t; i++) {
				let c = e + i * 64, l = a + i * 32, u = o + i * 8, d = new Float32Array(128);
				for (let e = 0; e < 32; e++) {
					let t = e >> 4, n = this.view.getUint8(c + e), r = this.view.getUint8(c + e + 32), i = this.view.getUint8(l + e), a = (n & 15 | (i >> 0 & 3) << 4) - 32, o = (r & 15 | (i >> 2 & 3) << 4) - 32, f = (n >> 4 | (i >> 4 & 3) << 4) - 32, p = (r >> 4 | (i >> 6 & 3) << 4) - 32, m = this.view.getInt8(u + t + 0), h = this.view.getInt8(u + t + 2), g = this.view.getInt8(u + t + 4), _ = this.view.getInt8(u + t + 6);
					d[e] = s * m * a, d[e + 32] = s * h * o, d[e + 64] = s * g * f, d[e + 96] = s * _ * p;
				}
				let f = Math.min(128, t - r);
				for (let e = 0; e < f; e++) n[r++] = d[e];
			}
			i += 210;
		}
		return n;
	}
	getTensorData(e, t) {
		let n = t + Number(e.offset), r = Number(e.dims.reduce((e, t) => e * t, 1n));
		if (e.type === 0) return new Float32Array(this.buffer, n, r);
		if (e.type === 1) return this.dequantizeF16(n, r);
		if (e.type === 8) return this.dequantizeQ8_0(n, r);
		if (e.type === 12) return this.dequantizeQ4_K(n, r);
		if (e.type === 13) return this.dequantizeQ5_K(n, r);
		if (e.type === 14) return this.dequantizeQ6_K(n, r);
		if (e.type === 30) return this.dequantizeBF16(n, r);
		throw Error(`Unsupported tensor type: ${e.type}`);
	}
};
function n(e) {
	let t = Number(e.dims.reduce((e, t) => e * t, 1n));
	if (e.type === 0) return t * 4;
	if (e.type === 1) return t * 2;
	if (e.type === 8) return t / 32 * 34;
	if (e.type === 12) return t / 256 * 144;
	if (e.type === 13) return t / 256 * 176;
	if (e.type === 14) return t / 256 * 210;
	if (e.type === 30) return t * 2;
	throw Error(`Unknown tensor type: ${e.type}`);
}
var r = /* @__PURE__ */ new ArrayBuffer(4), i = new Float32Array(r), a = new Uint32Array(r);
function o(e) {
	i[0] = e;
	let t = a[0], n = t >>> 16 & 32768, r = t >>> 23 & 255, o = t & 8388607;
	if (r === 255) return n | 31744 | (o ? 512 : 0);
	if (r === 0) return n;
	let s = r - 127 + 15;
	return s >= 31 ? n | 31744 : s <= 0 ? s < -10 ? n : n | (o | 8388608) >>> 1 - s + 13 : n | s << 10 | o >>> 13;
}
function s(e) {
	let t = globalThis.Float16Array;
	if (t) {
		let n = new t(e.length);
		return n.set(e), new Uint16Array(n.buffer);
	}
	let n = new Uint16Array(e.length), r = /* @__PURE__ */ new ArrayBuffer(4), i = new Float32Array(r), a = new Uint32Array(r);
	for (let t = 0; t < e.length; t++) {
		i[0] = e[t];
		let r = a[0], o = r >>> 16 & 32768, s = r >>> 23 & 255, c = r & 8388607;
		if (s === 255) {
			n[t] = o | 31744 | (c ? 512 : 0);
			continue;
		}
		if (s === 0) {
			n[t] = o;
			continue;
		}
		let l = s - 127 + 15;
		if (l >= 31) n[t] = o | 31744;
		else if (l <= 0) {
			if (l < -10) {
				n[t] = o;
				continue;
			}
			n[t] = o | (c | 8388608) >>> 1 - l + 13;
		} else n[t] = o | l << 10 | c >>> 13;
	}
	return n;
}
function c(e, t) {
	let n = e.kv.get(t);
	if (!n) return null;
	let r = n.value;
	return r == null || typeof r == "object" && "length" in r ? null : Number(r);
}
function l(e, t) {
	let n = e.kv.get(t);
	return !n || n.type !== "array" ? null : n.value.map((e) => e.value);
}
//#endregion
//#region src/tokenizer.ts
var u = {
	"<start_of_turn>": 105,
	"<end_of_turn>": 106,
	"<eos>": 1,
	"<bos>": 2
}, d = [
	"<start_function_declaration>",
	"<end_function_declaration>",
	"<start_function_call>",
	"<end_function_call>",
	"<start_function_response>",
	"<end_function_response>",
	"<escape>"
], f = [
	"<|im_start|>",
	"<|im_end|>",
	"<|endoftext|>",
	"<|object_ref_start|>",
	"<|object_ref_end|>",
	"<|box_start|>",
	"<|box_end|>",
	"<|quad_start|>",
	"<|quad_end|>",
	"<|vision_start|>",
	"<|vision_end|>",
	"<|vision_pad|>",
	"<|image_pad|>",
	"<|video_pad|>",
	"<tool_call>",
	"</tool_call>",
	"<|fim_prefix|>",
	"<|fim_middle|>",
	"<|fim_suffix|>",
	"<|fim_pad|>",
	"<|repo_name|>",
	"<|file_sep|>"
];
function p() {
	let e = [];
	for (let t = 33; t <= 126; t++) e.push(t);
	for (let t = 161; t <= 172; t++) e.push(t);
	for (let t = 174; t <= 255; t++) e.push(t);
	let t = e.slice(), n = 0;
	for (let r = 0; r < 256; r++) e.includes(r) || (e.push(r), t.push(256 + n), n++);
	let r = /* @__PURE__ */ new Map();
	for (let n = 0; n < e.length; n++) r.set(e[n], String.fromCodePoint(t[n]));
	return r;
}
var m = class {
	vocab = [];
	vocabByLength = [];
	tokenByText = /* @__PURE__ */ new Map();
	maxTokenLen = 0;
	specialTokens = { ...u };
	funcTokens = {};
	specialPatternRegex = /\\<start_of_turn\\>|\\<end_of_turn\\>|\\<eos\\>|\\<bos\\>/g;
	mode = "spm";
	byteEncoder = /* @__PURE__ */ new Map();
	byteDecoder = /* @__PURE__ */ new Map();
	bpeRanks = /* @__PURE__ */ new Map();
	addBos = !1;
	bpePat = /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
	extractFromGGUF(e) {
		let t = l(e, "tokenizer.ggml.tokens");
		if (!t) throw Error("No tokenizer found in GGUF metadata");
		this.vocab = t, this.vocabByLength = [];
		for (let e = 0; e < this.vocab.length; e++) this.vocab[e] && this.vocab[e].length > 0 && this.vocabByLength.push([e, this.vocab[e]]);
		this.buildTokenIndex();
		let n = e.kv.get("tokenizer.ggml.model")?.value, r = l(e, "tokenizer.ggml.merges");
		if (n === "gpt2" && r && r.length > 0) {
			this.mode = "bpe", this.byteEncoder = p();
			for (let [e, t] of this.byteEncoder) this.byteDecoder.set(t, e);
			for (let e = 0; e < r.length; e++) {
				let t = r[e].indexOf(" ");
				t <= 0 || this.bpeRanks.set(r[e].slice(0, t) + "" + r[e].slice(t + 1), e);
			}
			this.addBos = e.kv.get("tokenizer.ggml.add_bos_token")?.value === !0, this.initBpeSpecials();
		} else this.mode = "spm", this.initFunctionTokens();
	}
	buildTokenIndex() {
		this.tokenByText = /* @__PURE__ */ new Map(), this.maxTokenLen = 0;
		for (let e = 0; e < this.vocab.length; e++) {
			let t = this.vocab[e];
			t && t.length > 0 && (this.tokenByText.has(t) || this.tokenByText.set(t, e), t.length > this.maxTokenLen && (this.maxTokenLen = t.length));
		}
	}
	initFunctionTokens() {
		for (let e of d) {
			let t = this.tokenByText.get(e);
			t !== void 0 && (this.funcTokens[e] = t, this.specialTokens[e] = t);
		}
		this.rebuildSpecialPattern();
	}
	initBpeSpecials() {
		this.specialTokens = {};
		for (let e of f) {
			let t = this.tokenByText.get(e);
			t !== void 0 && (this.specialTokens[e] = t);
		}
		this.rebuildSpecialPattern();
	}
	rebuildSpecialPattern() {
		let e = Object.keys(this.specialTokens);
		if (e.length === 0) {
			this.specialPatternRegex = /(?!)/g;
			return;
		}
		e.sort((e, t) => t.length - e.length);
		let t = e.map((e) => e.replace(/[.*+?^${}()|[\]\\<>]/g, "\\$&")).join("|");
		this.specialPatternRegex = new RegExp(t, "g");
	}
	encodeSegment(e, t = !0) {
		let n = [], r = e.replace(/ /g, "▁");
		for (t && (r = "▁" + r); r.length > 0;) {
			let e = 0, t = -1, i = Math.min(r.length, this.maxTokenLen);
			for (let n = i; n >= 1; n--) {
				let i = this.tokenByText.get(r.substring(0, n));
				if (i !== void 0) {
					e = n, t = i;
					break;
				}
			}
			e === 0 ? r = r.slice(1) : (n.push(t), r = r.slice(e));
		}
		return n;
	}
	bpe(e) {
		let t = Array.from(e);
		if (t.length <= 1) return t;
		for (;;) {
			let e = Infinity, n = -1;
			for (let r = 0; r < t.length - 1; r++) {
				let i = this.bpeRanks.get(t[r] + "" + t[r + 1]);
				i !== void 0 && i < e && (e = i, n = r);
			}
			if (n < 0) break;
			t = t.slice(0, n).concat(t[n] + t[n + 1], t.slice(n + 2));
		}
		return t;
	}
	encodeBpeChunk(e) {
		let t = [], n = new TextEncoder(), r;
		for (this.bpePat.lastIndex = 0; (r = this.bpePat.exec(e)) !== null;) {
			let e = n.encode(r[0]), i = "";
			for (let t of e) i += this.byteEncoder.get(t);
			for (let e of this.bpe(i)) {
				let n = this.tokenByText.get(e);
				if (n !== void 0) t.push(n);
				else for (let n of e) {
					let e = this.tokenByText.get(n);
					e !== void 0 && t.push(e);
				}
			}
		}
		return t;
	}
	encodeBpe(e) {
		let t = [];
		if (this.addBos) {
			let e = this.specialTokens["<|endoftext|>"];
			e !== void 0 && t.push(e);
		}
		let n = new RegExp(this.specialPatternRegex.source, "g"), r = 0, i;
		for (; (i = n.exec(e)) !== null;) i.index > r && t.push(...this.encodeBpeChunk(e.slice(r, i.index))), t.push(this.specialTokens[i[0]]), r = i.index + i[0].length;
		return r < e.length && t.push(...this.encodeBpeChunk(e.slice(r))), t;
	}
	encode(e) {
		if (this.mode === "bpe") return this.encodeBpe(e);
		let t = [2], n = new RegExp(this.specialPatternRegex.source, "g"), r = 0, i, a = !1;
		for (; (i = n.exec(e)) !== null;) {
			let n = e.slice(r, i.index);
			n.length > 0 && t.push(...this.encodeSegment(n, !a)), t.push(this.specialTokens[i[0]]), r = i.index + i[0].length, a = !0;
		}
		let o = e.slice(r);
		return o.length > 0 && t.push(...this.encodeSegment(o, !a)), t;
	}
	decodeToken(e) {
		return e < this.vocab.length && this.vocab[e] ? this.mode === "bpe" ? this.decodeTokens([e]) : this.vocab[e].replace(/▁/g, " ") : `<unk:${e}>`;
	}
	decodeTokens(e) {
		if (this.mode === "bpe") {
			let t = [];
			for (let n of e) {
				let e = n < this.vocab.length ? this.vocab[n] : void 0;
				if (e) for (let n of e) {
					let e = this.byteDecoder.get(n);
					if (e !== void 0) t.push(e);
					else for (let e of new TextEncoder().encode(n)) t.push(e);
				}
			}
			return new TextDecoder().decode(Uint8Array.from(t));
		}
		let t = "";
		for (let n of e) t += this.decodeToken(n);
		return t;
	}
};
//#endregion
//#region src/conversation.ts
function h(e, t, n) {
	let r = null;
	try {
		if (t) {
			let e = JSON.parse(t);
			Array.isArray(e) && e.length > 0 && (r = e);
		}
	} catch {
		r = null;
	}
	let i = (n ?? "").trim(), a = "";
	if (i && (a += i + "\n"), r) {
		let e = "";
		for (let t of r) {
			if (e += `<start_function_declaration>declaration:${t.name}{`, e += `description:<escape>${t.description}<escape>`, t.parameters) {
				e += ",parameters:{properties:{";
				let n = Object.entries(t.parameters.properties || {});
				e += n.map(([e, t]) => {
					let n = `${e}:{description:<escape>${t.description}<escape>,type:<escape>${t.type}<escape>`;
					return t.enum && (n += `,enum:[${t.enum.map((e) => `<escape>${e}<escape>`).join(",")}]`), n += "}", n;
				}).join(","), e += "}", t.parameters.required && (e += `,required:[${t.parameters.required.map((e) => `<escape>${e}<escape>`).join(",")}]`), e += `,type:<escape>${t.parameters.type}<escape>`, e += "}";
			}
			e += "}<end_function_declaration>";
		}
		a += `You are a model that can do function calling with the following functions\n${e}\n`;
	}
	let o = "";
	a && (o += `<start_of_turn>developer\n${a.trimEnd()}<end_of_turn>\n`);
	for (let t of e) o += `<start_of_turn>${t.role}\n${t.text}<end_of_turn>\n`;
	return o += "<start_of_turn>model\n", o;
}
//#endregion
//#region src/config/gemma4-e2b.ts
function g() {
	let e = Array(35).fill(!0);
	for (let t of [
		4,
		9,
		14,
		19,
		24,
		29,
		34
	]) e[t] = !1;
	let t = Array(35);
	for (let e = 0; e < 35; e++) t[e] = e < 15 ? 6144 : 12288;
	let n = {
		hidden_size: 1536,
		q_dim: 8 * 512,
		kv_dim: 1 * 512,
		num_q_heads: 8,
		num_kv_heads: 1,
		head_dim: 256,
		intermediate_size: 12288,
		vocab_size: 262144,
		num_layers: 35,
		context_length: 2048,
		rms_norm_eps: 1e-6,
		rope_theta_global: 1e6,
		rope_theta_swa: 1e4,
		head_dim_local: 256,
		head_dim_global: 512,
		sliding_window: 512,
		attention_is_sliding: e,
		intermediate_sizes: t,
		num_unshared_layers: 15,
		kv_producer_for_layer: [],
		per_layer_input_dim: 256,
		final_logit_softcapping: 30
	};
	return n.kv_producer_for_layer = v(n), n;
}
function _(e, t) {
	if (!(e.kv.has("gemma4.block_count") || e.kv.has("gemma4.embedding_length"))) throw Error("Not a Gemma 4 GGUF — no `gemma4.*` metadata keys found.");
	let n = g(), r = c(e, "gemma4.embedding_length");
	r !== null && (n.hidden_size = r);
	let i = c(e, "gemma4.block_count");
	i !== null && (n.num_layers = i);
	let a = c(e, "gemma4.attention.head_count");
	a !== null && (n.num_q_heads = a);
	let o = c(e, "gemma4.attention.head_count_kv");
	o !== null && (n.num_kv_heads = o);
	let s = t ?? 2048, u = c(e, "gemma4.context_length");
	u !== null && (n.context_length = Math.min(u, s));
	let d = l(e, "gemma4.feed_forward_length");
	if (d) {
		let e = d.map((e) => Number(e));
		n.intermediate_sizes = e, n.intermediate_size = Math.max(...e);
	} else {
		let t = c(e, "gemma4.feed_forward_length");
		t !== null && (n.intermediate_size = t, n.intermediate_sizes = Array(n.num_layers).fill(t));
	}
	let f = c(e, "gemma4.attention.key_length"), p = c(e, "gemma4.attention.key_length_swa");
	f !== null && (n.head_dim_global = f), p !== null && (n.head_dim_local = p), n.head_dim = n.head_dim_local;
	let m = c(e, "gemma4.rope.freq_base"), h = c(e, "gemma4.rope.freq_base_swa");
	m !== null && (n.rope_theta_global = m), h !== null && (n.rope_theta_swa = h);
	let _ = c(e, "gemma4.attention.sliding_window");
	_ !== null && (n.sliding_window = _);
	let y = l(e, "gemma4.attention.sliding_window_pattern");
	y && (n.attention_is_sliding = y.map((e) => !!e));
	let b = c(e, "gemma4.embedding_length_per_layer_input");
	b !== null && (n.per_layer_input_dim = b);
	let x = c(e, "gemma4.final_logit_softcapping");
	x !== null && (n.final_logit_softcapping = x);
	let S = c(e, "gemma4.attention.shared_kv_layers");
	S !== null && (n.num_unshared_layers = n.num_layers - S);
	let C = Math.max(n.head_dim_local, n.head_dim_global);
	return n.q_dim = n.num_q_heads * C, n.kv_dim = n.num_kv_heads * C, n.kv_producer_for_layer = v(n), n;
}
function v(e) {
	let t = e.num_layers, n = e.num_unshared_layers, r = Array(t), i = -1, a = -1;
	for (let t = 0; t < n; t++) e.attention_is_sliding[t] ? i = t : a = t;
	for (let o = 0; o < t; o++) o < n ? r[o] = o : r[o] = e.attention_is_sliding[o] ? i : a;
	return r;
}
//#endregion
//#region src/config/qwen3-4b.ts
function y() {
	let e = 9728;
	return {
		hidden_size: 2560,
		q_dim: 4096,
		kv_dim: 1024,
		num_q_heads: 32,
		num_kv_heads: 8,
		head_dim: 128,
		intermediate_size: e,
		vocab_size: 151936,
		num_layers: 36,
		context_length: 2048,
		rms_norm_eps: 1e-6,
		rope_theta_global: 1e6,
		rope_theta_swa: 1e6,
		head_dim_local: 128,
		head_dim_global: 128,
		sliding_window: 0,
		attention_is_sliding: Array(36).fill(!1),
		intermediate_sizes: Array(36).fill(e),
		num_unshared_layers: 36,
		kv_producer_for_layer: Array.from({ length: 36 }, (e, t) => t),
		per_layer_input_dim: 0,
		final_logit_softcapping: 0,
		arch: "qwen3",
		ffn_activation: "silu",
		v_norm: !1,
		post_attn_norm: !1,
		post_ffn_norm: !1,
		embedding_scale: 1
	};
}
function b(e, t) {
	if (!(e.kv.has("qwen3.block_count") || e.kv.has("qwen3.embedding_length"))) throw Error("Not a Qwen3 GGUF — no `qwen3.*` metadata keys found.");
	let n = y(), r = c(e, "qwen3.embedding_length");
	r !== null && (n.hidden_size = r);
	let i = c(e, "qwen3.block_count");
	i !== null && (n.num_layers = i);
	let a = c(e, "qwen3.attention.head_count");
	a !== null && (n.num_q_heads = a);
	let o = c(e, "qwen3.attention.head_count_kv");
	o !== null && (n.num_kv_heads = o);
	let s = c(e, "qwen3.attention.key_length");
	s !== null && (n.head_dim = s, n.head_dim_local = s, n.head_dim_global = s);
	let l = c(e, "qwen3.feed_forward_length");
	l !== null && (n.intermediate_size = l);
	let u = c(e, "qwen3.attention.layer_norm_rms_epsilon");
	u !== null && (n.rms_norm_eps = u);
	let d = c(e, "qwen3.rope.freq_base");
	d !== null && (n.rope_theta_global = d, n.rope_theta_swa = d);
	let f = t ?? 2048, p = c(e, "qwen3.context_length");
	p !== null && (n.context_length = Math.min(p, f));
	let m = c(e, "qwen3.vocab_size");
	return m !== null && (n.vocab_size = m), n.lm_head_tensor = e.tensors.some((e) => e.name === "output.weight") ? "output" : void 0, n.q_dim = n.num_q_heads * n.head_dim, n.kv_dim = n.num_kv_heads * n.head_dim, n.attention_is_sliding = Array(n.num_layers).fill(!1), n.intermediate_sizes = Array(n.num_layers).fill(n.intermediate_size), n.num_unshared_layers = n.num_layers, n.kv_producer_for_layer = Array.from({ length: n.num_layers }, (e, t) => t), n;
}
//#endregion
//#region src/tuning/profile.ts
function x(e, t) {
	return e.matmul.rowsPerWorkgroupByKernel?.[t] ?? e.matmul.defaultRowsPerWorkgroup;
}
function S(e, t) {
	return {
		id: t.id ?? e.id,
		description: t.description ?? e.description,
		verified: t.verified ?? e.verified,
		notes: t.notes ?? e.notes,
		matmul: {
			workgroupSize: t.matmul?.workgroupSize ?? e.matmul.workgroupSize,
			defaultRowsPerWorkgroup: t.matmul?.defaultRowsPerWorkgroup ?? e.matmul.defaultRowsPerWorkgroup,
			rowsPerWorkgroupByKernel: {
				...e.matmul.rowsPerWorkgroupByKernel ?? {},
				...t.matmul?.rowsPerWorkgroupByKernel ?? {}
			}
		},
		pipeline: {
			decodeDepth: t.pipeline?.decodeDepth ?? e.pipeline.decodeDepth,
			greedyFastPath: t.pipeline?.greedyFastPath ?? e.pipeline.greedyFastPath
		},
		features: {
			shaderF16Required: t.features?.shaderF16Required ?? e.features.shaderF16Required,
			subgroups: t.features?.subgroups ?? e.features.subgroups,
			subgroupMatrix: t.features?.subgroupMatrix ?? e.features.subgroupMatrix
		}
	};
}
//#endregion
//#region src/tuning/devices.ts
var C = {
	id: "nvidia-blackwell",
	description: "NVIDIA Blackwell (RTX 50-series). F16 matmul, MR4 FFN, depth-2 pipelined decode.",
	verified: !0,
	notes: "Measured ~127 short / ~116 long tps on RTX 5090 / Chrome stable (prompt \"Hello, how are you?\" maxTokens=64; raven/crow maxTokens=200). MR4 wins ~8% on ffn.linearGateUp; pipelined decode hides ~2.7 ms/token of submit→callback slack.",
	matmul: {
		workgroupSize: 256,
		defaultRowsPerWorkgroup: 1,
		rowsPerWorkgroupByKernel: { "ffn.linearGateUp": 4 }
	},
	pipeline: {
		decodeDepth: 2,
		greedyFastPath: !0
	},
	features: {
		shaderF16Required: !0,
		subgroups: "avoid",
		subgroupMatrix: "avoid"
	}
}, w = {
	id: "apple-m-series",
	description: "Apple M-series (M1/M2/M3). Conservative starting defaults; awaits calibration.",
	verified: !1,
	notes: "Starting defaults based on Apple GPU architectural priors: prefers single-row matmul, serial decode is robust. Run a calibration sweep and update this profile — file a device-report issue with results.",
	matmul: {
		workgroupSize: 256,
		defaultRowsPerWorkgroup: 1
	},
	pipeline: {
		decodeDepth: 2,
		greedyFastPath: !0
	},
	features: {
		shaderF16Required: !0,
		subgroups: "avoid",
		subgroupMatrix: "avoid"
	}
}, T = {
	id: "generic",
	description: "Conservative portable defaults. Works on any WebGPU + shader-f16 device.",
	verified: !1,
	notes: "Baseline portable defaults. Expect ~20–30% lower tps vs a tuned vendor profile (measured ~27% short on RTX 5090 with both MR4 + depth-2 pipelined decode off). Use as the starting point when calibrating new hardware.",
	matmul: {
		workgroupSize: 256,
		defaultRowsPerWorkgroup: 1
	},
	pipeline: {
		decodeDepth: 1,
		greedyFastPath: !1
	},
	features: {
		shaderF16Required: !0,
		subgroups: "avoid",
		subgroupMatrix: "avoid"
	}
}, E = {
	"nvidia-blackwell": C,
	"apple-m-series": w,
	generic: T
};
//#endregion
//#region src/tuning/detect.ts
function D(e) {
	if (typeof e != "object" || !e) return !1;
	let t = e;
	return typeof t.id == "string" && typeof t.description == "string" && typeof t.verified == "boolean" && typeof t.matmul == "object" && typeof t.pipeline == "object" && typeof t.features == "object";
}
function O(e, t) {
	if (D(t)) return {
		profile: t,
		reason: `explicit profile object: ${t.id}`
	};
	if (typeof t == "string") {
		let e = E[t];
		if (!e) {
			let e = Object.keys(E).join(", ");
			throw Error(`unknown tuning profile id "${t}" (available: ${e})`);
		}
		return {
			profile: e,
			reason: `explicit override: "${t}"`
		};
	}
	if (typeof t == "object" && t) {
		let n = k(e);
		return {
			profile: S(n.profile, t),
			reason: `${n.reason}; then applied caller overrides`
		};
	}
	return k(e);
}
function k(e) {
	let t = e.info, n = `${(t.vendor ?? "").toLowerCase()} ${(t.architecture ?? "").toLowerCase()} ${(t.description ?? "").toLowerCase()}`;
	return n.includes("nvidia") ? {
		profile: E["nvidia-blackwell"],
		reason: `vendor match: nvidia → ${E["nvidia-blackwell"].id}`
	} : n.includes("apple") ? {
		profile: E["apple-m-series"],
		reason: `vendor match: apple → ${E["apple-m-series"].id}`
	} : {
		profile: E.generic,
		reason: `no vendor match (vendor="${t.vendor ?? ""}", architecture="${t.architecture ?? ""}") → generic`
	};
}
//#endregion
//#region src/ranged-reader.ts
var A = class {
	constructor(e, t = {}) {
		this.url = e, this.options = t;
	}
	async readAll(e, t) {
		let n = new Uint8Array(t), r = 0;
		for await (let i of this.stream(e, t)) n.set(i, r), r += i.byteLength;
		return n;
	}
	async *stream(e, t) {
		let n = this.options.noProgressTimeoutMs ?? 3e4, r = this.options.maxAttempts ?? 4, i = this.options.signal, a = this.options.onEvent, o = 0;
		for (let s = 0; s < r; s++) {
			let c = new AbortController(), l = () => {
				c.abort(i?.reason ?? new DOMException("Aborted", "AbortError"));
			};
			i && (i.aborted ? l() : i.addEventListener("abort", l, { once: !0 }));
			let u = null, d = () => {
				u && clearTimeout(u), u = setTimeout(() => {
					c.abort(/* @__PURE__ */ Error(`no progress for ${n}ms`));
				}, n);
			}, f = () => {
				u &&= (clearTimeout(u), null);
			};
			try {
				let n = e + o, r = e + t - 1, s = await fetch(this.url, {
					headers: { Range: `bytes=${n}-${r}` },
					signal: c.signal
				});
				if (s.status !== 206) throw Error(`expected HTTP 206 for Range ${n}-${r}, got ${s.status}`);
				let u = s.body.getReader();
				for (;;) {
					d();
					let { done: e, value: n } = await u.read();
					if (f(), e) break;
					o += n.byteLength, a?.({
						type: "chunk",
						bytesRead: o,
						totalSize: t
					}), yield n;
				}
				if (o < t) throw Error(`premature EOF: ${o}/${t} bytes`);
				i?.removeEventListener("abort", l);
				return;
			} catch (e) {
				if (f(), i?.removeEventListener("abort", l), i?.aborted || s >= r - 1) throw e;
				a?.({
					type: "retry",
					attempt: s + 1,
					reason: e instanceof Error ? e.message : String(e),
					bytesReadBeforeRetry: o
				});
				let t = 500 * 2 ** s;
				await new Promise((e) => setTimeout(e, t));
			}
		}
	}
};
//#endregion
//#region src/types.ts
function j(e, t) {
	return t.attention_is_sliding[e];
}
function M(e, t) {
	return j(e, t) ? t.rope_theta_swa : t.rope_theta_global;
}
function N(e, t) {
	return j(e, t) ? t.head_dim_local : t.head_dim_global;
}
function P(e, t) {
	return t.intermediate_sizes[e] ?? t.intermediate_size;
}
function F(e, t) {
	return t.kv_producer_for_layer[e] === e;
}
//#endregion
//#region src/engine.ts
var I = { e2b: "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf" }, L = 256 * 1024 * 1024, R = globalThis.Float16Array, z = [
	"attn_q",
	"attn_k",
	"attn_v",
	"attn_output",
	"ffn_gate",
	"ffn_up",
	"ffn_down"
], B = [
	"attn_norm",
	"ffn_norm",
	"attn_q_norm",
	"attn_k_norm",
	"post_attention_norm",
	"post_ffw_norm",
	"inp_gate",
	"proj",
	"post_norm",
	"layer_output_scale"
], V = [
	"per_layer_proj_norm",
	"rope_freqs",
	"per_layer_model_proj"
], H = class r {
	config;
	tuning;
	tuningReason;
	adapter;
	device;
	pipelines;
	mr4ForFfn;
	matmulRowsPerWg = 1;
	weightQuant = "f16";
	modelBuffers;
	workBuffers;
	uniformBuffers;
	kvCaches;
	bindGroupCache;
	tokenizer;
	conversationHistory = [];
	kvPosition = 0;
	_systemPrompt = null;
	_lastGenerateStats = null;
	onProgress;
	deviceLost = !1;
	profileCapability = {
		timestampQuerySupported: !1,
		timestampQueryOnAdapterButUnrequested: !1,
		querySlots: 0
	};
	profileActive = !1;
	profileCapturing = !1;
	profileQuerySet = null;
	profileResolveBuf = null;
	profileStagingBuf = null;
	profileSlotCursor = 0;
	profilePassLabels = [];
	profileSamples = /* @__PURE__ */ new Map();
	profileForwardTotals_ns = [];
	profileOverflow = !1;
	static PROFILE_QUERY_SLOTS = 2048;
	cpuProfileActive = !1;
	cpuProfileCapturing = !1;
	cpuPhaseSamples = /* @__PURE__ */ new Map();
	cpuForwardSamples_ms = [];
	cpuForwardWriteBufferCounts = [];
	cpuForwardPassEncodeCounts = [];
	cpuWbCountThisForward = 0;
	cpuPassCountThisForward = 0;
	cpuPrevForwardEnd_ms = 0;
	argmaxReadbackIdx = 0;
	modelFile;
	ggufTensors;
	ggufDataOffset;
	_loadTimings = null;
	get loadTimings() {
		return this._loadTimings;
	}
	constructor(e) {
		this.config = g(), e.contextLength && (this.config.context_length = e.contextLength), this.onProgress = e.onProgress;
	}
	async init(e) {
		let n = e.model || "e2b", r = I[n] || n, i = performance.now();
		this._loadTimings = {
			supportsRange: !1,
			modelFile: r,
			startedAt_ms: i,
			totalMs: 0,
			phases: {}
		};
		let a = i, o = (e) => {
			let t = performance.now();
			this._loadTimings.phases[e] = t - a, a = t;
		};
		await this.initWebGPU(), o("initWebGPU_ms");
		let s = O(this.adapter, e.tuning);
		this.tuning = s.profile, this.tuningReason = s.reason, this.mr4ForFfn = x(this.tuning, "ffn.linearGateUp") >= 4, this.weightQuant = e.weightQuant ?? "f16", this.matmulRowsPerWg = this.weightQuant === "q4k" || this.weightQuant === "q8" || this.mr4ForFfn ? 4 : 1, console.log(`[gemma4-webgpu] tuning: ${this.tuning.id} (${s.reason})${this.tuning.verified ? " — verified" : " — unverified"}`), this.reportProgress(0, 1, "Downloading header...", "downloading");
		let c = await fetch(r, {
			headers: { Range: `bytes=0-${20 * 1024 * 1024 - 1}` },
			signal: e.signal
		}), l = c.status === 206;
		if (this._loadTimings.supportsRange = l, l) {
			let n = new Uint8Array(await c.arrayBuffer());
			o("headerFetch_ms");
			let i = new t(n).parse();
			this.config = i.kv.has("qwen3.block_count") || i.kv.has("qwen3.embedding_length") ? b(i, e.contextLength) : _(i, e.contextLength), this.tokenizer = new m(), this.tokenizer.extractFromGGUF(i);
			let s = i.tensors, l = i.dataOffset;
			this.modelFile = r, this.ggufTensors = s, this.ggufDataOffset = l, o("ggufParse_ms"), this.createPipelines(), o("createPipelines_ms"), this.createUniformBuffers(), o("createUniformBuffers_ms"), await this.uploadWeightsStreaming(r, s, l, e.signal), a = performance.now();
		} else {
			let n = c.headers.get("content-length"), r = n ? parseInt(n) : 11e8, i = c.body.getReader(), a = new Uint8Array(r), s = 0;
			for (;;) {
				let { done: e, value: t } = await i.read();
				if (e) break;
				a.set(t, s), s += t.length, this.reportProgress(s, r, "Downloading model...");
			}
			o("fullBufferDownload_ms");
			let l = new t(a), u = l.parse();
			this.config = u.kv.has("qwen3.block_count") || u.kv.has("qwen3.embedding_length") ? b(u, e.contextLength) : _(u, e.contextLength), this.tokenizer = new m(), this.tokenizer.extractFromGGUF(u), this.ggufTensors = u.tensors, this.ggufDataOffset = u.dataOffset, o("ggufParse_ms"), this.createPipelines(), o("createPipelines_ms"), this.createUniformBuffers(), o("createUniformBuffers_ms"), await this.uploadWeightsFromBuffer(l, u), await this.device.queue.onSubmittedWorkDone(), o("fullBufferUpload_ms");
		}
		if (this.config.arch !== "qwen3" && this.config.per_layer_input_dim === 0) throw Error("per_layer_input_dim = 0 — this engine is Gemma-4-E2B-specific and requires PLE. Check that the GGUF is a Gemma 4 variant.");
		this.createWorkBuffers(), o("createWorkBuffers_ms"), this.createBindGroups(), o("createBindGroups_ms"), this._loadTimings.totalMs = performance.now() - i;
	}
	wb(e, t, n) {
		this.cpuProfileActive && this.cpuProfileCapturing && this.cpuWbCountThisForward++, this.device.queue.writeBuffer(e, t, n);
	}
	recordCpuPhase(e, t) {
		let n = this.cpuPhaseSamples.get(e);
		n || (n = [], this.cpuPhaseSamples.set(e, n)), n.push(t);
	}
	dispatchMatmul(e, t) {
		let n = 65535;
		if (t <= n) {
			e.dispatchWorkgroups(t, 1, 1);
			return;
		}
		let r = Math.min(n, Math.ceil(Math.sqrt(t))), i = Math.ceil(t / r);
		e.dispatchWorkgroups(r, i, 1);
	}
	dispatchMatmulRows(e, t) {
		this.dispatchMatmul(e, Math.ceil(t / this.matmulRowsPerWg));
	}
	reportProgress(e, t, n, r) {
		this.onProgress && this.onProgress({
			loaded: e,
			total: t,
			status: n,
			kind: r
		});
	}
	async initWebGPU() {
		if (!navigator.gpu) throw Error("WebGPU not supported");
		let e = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
		if (!e) throw Error("No WebGPU adapter found");
		this.adapter = e;
		let t = e.features.has("shader-f16"), n = e.features.has("timestamp-query"), i = [];
		t && i.push("shader-f16"), n && i.push("timestamp-query"), this.device = await e.requestDevice({
			requiredFeatures: i,
			requiredLimits: {
				maxStorageBufferBindingSize: e.limits.maxStorageBufferBindingSize,
				maxBufferSize: e.limits.maxBufferSize
			}
		});
		let a = this.device.features.has("timestamp-query");
		if (this.profileCapability = {
			timestampQuerySupported: a,
			timestampQueryOnAdapterButUnrequested: n && !a,
			querySlots: a ? r.PROFILE_QUERY_SLOTS : 0
		}, !t) throw Error("shader-f16 WebGPU feature is required (your adapter reports it unavailable)");
		this.device.lost.then((e) => {
			this.deviceLost = !0, console.error(`WebGPU device lost: ${e.message} (reason: ${e.reason})`);
		});
	}
	createPipelines() {
		this.pipelines = {};
		for (let [t, n] of Object.entries(e)) {
			let e = {
				module: this.device.createShaderModule({ code: n }),
				entryPoint: "main"
			};
			if (t === "matmulQuantMR4") {
				let t = x(this.tuning, "ffn.linearGateUp");
				e.constants = { R: t >= 4 ? t : 4 };
			}
			this.pipelines[t] = this.device.createComputePipeline({
				layout: "auto",
				compute: e
			});
		}
	}
	makeUniformMixed(e) {
		let t = Math.max(e.length * 4, 16), n = new ArrayBuffer(t), r = new Uint32Array(n), i = new Float32Array(n);
		for (let t = 0; t < e.length; t++) {
			let n = e[t];
			"u" in n ? r[t] = n.u : i[t] = n.f;
		}
		let a = this.device.createBuffer({
			size: t,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			mappedAtCreation: !0
		});
		return new Uint8Array(a.getMappedRange()).set(new Uint8Array(n)), a.unmap(), a;
	}
	createUniformBuffers() {
		let e = this.config, t = e.hidden_size, n = e.context_length, r = e.vocab_size, i = e.num_q_heads, a = e.num_kv_heads;
		this.uniformBuffers = {
			rmsNorm: this.makeUniformMixed([{ u: t }, { f: e.rms_norm_eps }]),
			sizeH: this.makeUniformMixed([{ u: t }]),
			embeddingLookup: this.makeUniformMixed([
				{ u: t },
				{ u: 0 },
				{ f: e.embedding_scale ?? Math.sqrt(t) }
			]),
			softmax: this.makeUniformMixed([{ u: i }, { u: 0 }]),
			linearQ8_V_H: this.makeUniformMixed([{ u: r }, { u: t }]),
			argmaxSize: this.makeUniformMixed([{ u: r }]),
			logitSoftcap: this.makeUniformMixed([{ u: r }, { f: e.final_logit_softcapping }]),
			plePmProjMM: this.makeUniformMixed([{ u: e.num_layers * e.per_layer_input_dim }, { u: t }]),
			pleInpGateMM: this.makeUniformMixed([{ u: e.per_layer_input_dim }, { u: t }]),
			plePostProjMM: this.makeUniformMixed([{ u: t }, { u: e.per_layer_input_dim }]),
			pleStage1: [],
			pleGeluMulParams: [],
			pleSkipScaleAdd: this.makeUniformMixed([{ u: t }]),
			perHeadRmsNormQ: [],
			perHeadRmsNormK: [],
			perHeadRmsNormV: [],
			linearQ8_Q_H: [],
			linearQ8_KV_H: [],
			linearQ8_H_Q: [],
			linearQ8_I_H: [],
			linearQ8_H_I: [],
			sizeI: [],
			kvCacheStore: [],
			attnScore: [],
			attnOutput: [],
			ropeQ: [],
			ropeK: [],
			fusedNormRopeQ: [],
			fusedNormRopeK: []
		};
		for (let r = 0; r < e.num_layers; r++) {
			let o = N(r, e), s = P(r, e), c = i * o, l = a * o, u = M(r, e);
			this.uniformBuffers.perHeadRmsNormQ.push(this.makeUniformMixed([
				{ u: i },
				{ u: o },
				{ f: e.rms_norm_eps },
				{ u: 0 }
			])), this.uniformBuffers.perHeadRmsNormK.push(this.makeUniformMixed([
				{ u: a },
				{ u: o },
				{ f: e.rms_norm_eps },
				{ u: 0 }
			])), this.uniformBuffers.perHeadRmsNormV.push(this.makeUniformMixed([
				{ u: a },
				{ u: o },
				{ f: e.rms_norm_eps },
				{ u: 0 }
			])), this.uniformBuffers.linearQ8_Q_H.push(this.makeUniformMixed([{ u: c }, { u: t }])), this.uniformBuffers.linearQ8_KV_H.push(this.makeUniformMixed([{ u: l }, { u: t }])), this.uniformBuffers.linearQ8_H_Q.push(this.makeUniformMixed([{ u: t }, { u: c }])), this.uniformBuffers.linearQ8_I_H.push(this.makeUniformMixed([{ u: s }, { u: t }])), this.uniformBuffers.linearQ8_H_I.push(this.makeUniformMixed([{ u: t }, { u: s }])), this.uniformBuffers.sizeI.push(this.makeUniformMixed([{ u: s }])), this.uniformBuffers.kvCacheStore.push(this.makeUniformMixed([
				{ u: a },
				{ u: o },
				{ u: 0 },
				{ u: n }
			]));
			let d = j(r, e) ? e.sliding_window : 0, f = e.arch === "qwen3" ? 1 / Math.sqrt(o) : 1;
			this.uniformBuffers.attnScore.push(this.makeUniformMixed([
				{ u: i },
				{ u: a },
				{ u: o },
				{ u: 0 },
				{ f },
				{ u: d }
			])), this.uniformBuffers.attnOutput.push(this.makeUniformMixed([
				{ u: i },
				{ u: a },
				{ u: o },
				{ u: 0 }
			]));
			let p = e.arch === "qwen3" ? 0 : +!j(r, e);
			this.uniformBuffers.ropeQ.push(this.makeUniformMixed([
				{ u: i },
				{ u: o },
				{ u: 0 },
				{ f: u }
			])), this.uniformBuffers.ropeK.push(this.makeUniformMixed([
				{ u: a },
				{ u: o },
				{ u: 0 },
				{ f: u }
			])), this.uniformBuffers.fusedNormRopeQ.push(this.makeUniformMixed([
				{ u: i },
				{ u: o },
				{ f: e.rms_norm_eps },
				{ u: 0 },
				{ f: u },
				{ u: p }
			])), this.uniformBuffers.fusedNormRopeK.push(this.makeUniformMixed([
				{ u: a },
				{ u: o },
				{ f: e.rms_norm_eps },
				{ u: 0 },
				{ f: u },
				{ u: p }
			])), this.uniformBuffers.pleStage1.push(this.makeUniformMixed([
				{ u: r },
				{ u: 0 },
				{ u: e.per_layer_input_dim },
				{ f: e.rms_norm_eps }
			])), this.uniformBuffers.pleGeluMulParams.push(this.makeUniformMixed([{ u: r * e.per_layer_input_dim }, { u: e.per_layer_input_dim }]));
		}
	}
	createF16Buffer(e, t = !1) {
		let n = e;
		e.byteLength & 3 && (n = new Uint16Array(e.byteLength + 3 >>> 1 & -2), n.set(e));
		let r = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | (t ? GPUBufferUsage.COPY_SRC : 0), i = this.device.createBuffer({
			size: n.byteLength,
			usage: r
		});
		return this.wb(i, 0, n), i;
	}
	createQ8Buffers(e, t, n) {
		let r = n / 32, i = new Uint8Array(t * n), a = new Uint16Array(t * r);
		for (let s = 0; s < t; s++) {
			let t = s * n, c = s * r;
			for (let n = 0; n < r; n++) {
				let r = t + n * 32, s = 0;
				for (let t = 0; t < 32; t++) {
					let n = Math.abs(e[r + t]);
					n > s && (s = n);
				}
				let l = s / 127, u = l > 0 ? 1 / l : 0;
				a[c + n] = o(l);
				for (let t = 0; t < 32; t++) {
					let n = Math.round(e[r + t] * u);
					n > 127 ? n = 127 : n < -127 && (n = -127), i[r + t] = n & 255;
				}
			}
		}
		let s = new Uint32Array(i.buffer), c = this.device.createBuffer({
			size: s.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		return this.wb(c, 0, s), {
			quants: c,
			scales: this.createF16Buffer(a)
		};
	}
	mmQ8(e, t, n, r, i) {
		return this.device.createBindGroup({
			layout: this.pipelines.matmulQ8.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: { buffer: e }
				},
				{
					binding: 1,
					resource: { buffer: t[n] }
				},
				{
					binding: 2,
					resource: { buffer: t[`${n}__s`] }
				},
				{
					binding: 3,
					resource: { buffer: r }
				},
				{
					binding: 4,
					resource: { buffer: i }
				}
			]
		});
	}
	createQ4KBuffers(e, t, n) {
		let r = n / 256, i = new Uint8Array(t * n), a = new Uint16Array(t * r * 16);
		for (let s = 0; s < t; s++) {
			let t = s * n, c = s * r * 16;
			for (let n = 0; n < r; n++) {
				let r = t + n * 256, s = c + n * 16;
				for (let t = 0; t < 8; t++) {
					let n = r + t * 32, c = Infinity, l = -Infinity;
					for (let t = 0; t < 32; t++) {
						let r = e[n + t];
						r < c && (c = r), r > l && (l = r);
					}
					let u = (l - c) / 15, d = u > 0 ? 1 / u : 0;
					a[s + t] = o(u), a[s + 8 + t] = o(c);
					for (let t = 0; t < 32; t++) {
						let r = Math.round((e[n + t] - c) * d);
						r > 15 ? r = 15 : r < 0 && (r = 0), i[n + t] = r;
					}
				}
			}
		}
		let s = new Uint32Array(t * n / 8);
		for (let e = 0; e < t * n; e++) s[e >> 3] |= (i[e] & 15) << (e & 7) * 4;
		let c = this.device.createBuffer({
			size: s.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
		});
		return this.wb(c, 0, s), {
			quants: c,
			meta: this.createF16Buffer(a)
		};
	}
	mmQuant(e, t, n, r, i) {
		let a = this.weightQuant === "q4k" ? this.pipelines.matmulQ4KMr : this.pipelines.matmulQ8Mr;
		return this.device.createBindGroup({
			layout: a.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: { buffer: e }
				},
				{
					binding: 1,
					resource: { buffer: t[n] }
				},
				{
					binding: 2,
					resource: { buffer: t[`${n}__s`] }
				},
				{
					binding: 3,
					resource: { buffer: r }
				},
				{
					binding: 4,
					resource: { buffer: i }
				}
			]
		});
	}
	tensorToF16(e, t, n) {
		if (t.type === 1) {
			let r = n + Number(t.offset), i = Number(t.dims.reduce((e, t) => e * t, 1n));
			return new Uint16Array(e.buffer.slice(r, r + i * 2));
		}
		let r = n + Number(t.offset), i = Number(t.dims.reduce((e, t) => e * t, 1n));
		return i > L ? this.dequantToF16Chunked(e, t, r, i) : s(e.getTensorData(t, n));
	}
	dequantToF16Chunked(e, t, n, r) {
		let i, a;
		switch (t.type) {
			case 12:
				i = 256, a = 144;
				break;
			case 13:
				i = 256, a = 176;
				break;
			case 14:
				i = 256, a = 210;
				break;
			case 8:
				i = 32, a = 34;
				break;
			default: return s(e.getTensorData({
				...t,
				offset: BigInt(n)
			}, 0));
		}
		let c = new Uint16Array(r), l = Math.max(i, Math.floor((1 << 24) / i) * i), u = 0, d = n;
		for (; u < r;) {
			let n = Math.min(l, r - u), s;
			switch (t.type) {
				case 12:
					s = e.dequantizeQ4_K(d, n);
					break;
				case 13:
					s = e.dequantizeQ5_K(d, n);
					break;
				case 14:
					s = e.dequantizeQ6_K(d, n);
					break;
				default:
					s = e.dequantizeQ8_0(d, n);
					break;
			}
			for (let e = 0; e < n; e++) c[u + e] = o(s[e]);
			u += n, d += n / i * a;
		}
		return c;
	}
	async uploadWeightsFromBuffer(e, t) {
		let n = t.tensors, r = t.dataOffset;
		this.modelBuffers = {
			tokenEmbed: null,
			layers: [],
			finalNorm: null,
			globals: {},
			perLayerEmbeddings: []
		};
		let i = (t, n = !1) => this.createF16Buffer(this.tensorToF16(e, t, r), n), a = n.find((e) => e.name === "token_embd.weight");
		a && (this.modelBuffers.tokenEmbed = i(a, !0));
		let o = n.find((e) => e.name === "output_norm.weight");
		o && (this.modelBuffers.finalNorm = i(o, !0));
		for (let e of [...V]) {
			let t = n.find((t) => t.name === e + ".weight");
			t && (this.modelBuffers.globals[e] = i(t, !0));
		}
		await this.uploadPerLayerEmbeddingsFromBuffer(e, n, r);
		for (let e = 0; e < this.config.num_layers; e++) {
			let t = `blk.${e}.`, r = {};
			for (let e of [...B, ...z]) {
				let a = n.find((n) => n.name === t + e + ".weight");
				a && (r[e] = i(a, !0));
			}
			this.modelBuffers.layers.push(r);
		}
	}
	async streamDecodePleQ5_K(e, n) {
		let r = this.config.num_layers, i = this.config.per_layer_input_dim, a = this.config.vocab_size;
		if (i !== 256) throw Error(`PLE streaming assumes per_layer_input_dim == 256 (got ${i})`);
		let o = a * i * 2;
		for (let e = 0; e < r; e++) {
			let e = this.device.createBuffer({
				size: o + 3 & -4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
			});
			this.modelBuffers.perLayerEmbeddings.push(e);
		}
		let c = [];
		for (let e = 0; e < r; e++) c.push(new Float32Array(512 * i));
		let l = r * 176, u = 512 * l, d = new Uint8Array(u), f = 0, p = 0, m = 0, h = a, g = (e) => {
			let n = new t(d.subarray(0, e * l));
			for (let t = 0; t < e; t++) {
				let e = t * i;
				for (let i = 0; i < r; i++) {
					let a = (t * r + i) * 176;
					n.decodeQ5_KBlock(a, c[i], e);
				}
			}
			for (let t = 0; t < r; t++) {
				let n = s(e === 512 ? c[t] : c[t].subarray(0, e * i));
				this.device.queue.writeBuffer(this.modelBuffers.perLayerEmbeddings[t], p * i * 2, n.buffer, 0, e * i * 2);
			}
			p += e, f = 0;
			let a = (p / h * 100).toFixed(1);
			this.reportProgress(p, h, `PLE pipeline: ${a}% (vocab ${p}/${h})`);
		};
		for await (let t of e) {
			m += t.byteLength;
			let e = 0;
			for (; e < t.byteLength;) {
				let n = u - f, r = Math.min(n, t.byteLength - e);
				d.set(t.subarray(e, e + r), f), f += r, e += r, f === u && g(512);
			}
		}
		if (m !== n) throw Error(`PLE streaming: received ${m} bytes but expected ${n}`);
		if (f > 0) {
			let e = h - p;
			if (e * l !== f) throw Error(`PLE streaming: accumulator has ${f} bytes but ${e} vocab entries remain (expected ${e * l})`);
			g(e);
		}
		if (p !== h) throw Error(`PLE streaming: processed ${p} vocab entries, expected ${h}`);
	}
	async uploadPerLayerEmbeddingsFromBuffer(e, t, r) {
		let i = t.find((e) => e.name === "per_layer_token_embd.weight");
		if (!i) return;
		let a = r + Number(i.offset), o = n(i), s = new Uint8Array(e.buffer, a, o);
		async function* c() {
			yield s;
		}
		await this.streamDecodePleQ5_K(c(), o);
	}
	async uploadWeightsStreaming(e, r, i, a) {
		this.modelBuffers = {
			tokenEmbed: null,
			layers: [],
			finalNorm: null,
			globals: {},
			perLayerEmbeddings: []
		};
		let o = new A(e, {
			noProgressTimeoutMs: 3e4,
			maxAttempts: 4,
			signal: a,
			onEvent: (e) => {
				e.type === "retry" && (console.warn(`[gemma4-webgpu] Range reader retry #${e.attempt} after ${e.bytesReadBeforeRetry} bytes: ${e.reason}`), this.reportProgress(p, m, `Connection stalled, retrying (attempt ${e.attempt})…`, "retrying"));
			}
		}), c = (e, t) => o.readAll(e, t), l = (e, n, r) => {
			if (r.type === 1) {
				let t = Number(r.dims.reduce((e, t) => e * t, 1n));
				return new Uint16Array(e.buffer.slice(e.byteOffset + n, e.byteOffset + n + t * 2));
			}
			let i = new t(e), a = Number(r.dims.reduce((e, t) => e * t, 1n));
			return a > L ? this.dequantToF16Chunked(i, r, n, a) : s(i.getTensorData({
				...r,
				offset: BigInt(n)
			}, 0));
		}, u = (e, t, n, r = !1) => {
			let i = l(e, t, n);
			return {
				buf: this.createF16Buffer(i, r),
				byteLength: i.byteLength
			};
		}, d = (e, n, r) => {
			let i = new t(e).getTensorData({
				...r,
				offset: BigInt(n)
			}, 0), a = Number(r.dims[0]), o = Number(r.dims[1]);
			return this.createQ8Buffers(i, o, a);
		}, f = (e, n, r) => {
			let i = new t(e).getTensorData({
				...r,
				offset: BigInt(n)
			}, 0), a = Number(r.dims[0]), o = Number(r.dims[1]);
			return this.createQ4KBuffers(i, o, a);
		}, p = 0, m = r.reduce((e, t) => e + n(t), 0), h = 0, g = 0, _ = 0, v = async (e, t = !1) => {
			let a = r.find((t) => t.name === e + ".weight");
			if (!a) return null;
			let o = i + Number(a.offset), s = n(a), l = u(await c(o, s), 0, a, t);
			return p += s, l.buf;
		};
		if (this.modelBuffers.tokenEmbed = await v("token_embd", !0), this.reportProgress(p, m, "Streaming weights to GPU..."), this.modelBuffers.finalNorm = await v("output_norm", !0), this.config.lm_head_tensor) {
			let e = await v(this.config.lm_head_tensor, !0);
			e && (this.modelBuffers.globals.lm_head = e);
		}
		for (let e of V) {
			let t = await v(e, !0);
			t && (this.modelBuffers.globals[e] = t, this.reportProgress(p, m, `Loaded ${e}`));
		}
		let y = r.find((e) => e.name === "per_layer_token_embd.weight");
		if (y) {
			let e = i + Number(y.offset), t = n(y);
			this.reportProgress(p, m, `Streaming per_layer_token_embd (~${(t / 1e9).toFixed(2)} GB Q5_K)…`), this._loadTimings && (this._loadTimings.pleBytes = t);
			let r = performance.now();
			await this.streamDecodePleQ5_K(o.stream(e, t), t), this._loadTimings && (this._loadTimings.phases.plePipeline_ms = performance.now() - r), p += t;
		}
		let b = this.config.num_layers;
		this.modelBuffers.layers = Array(b);
		let x = Array(b).fill(null), S = 0, C = async (e) => {
			let t = `blk.${e}.`, a = r.filter((e) => e.name.startsWith(t));
			if (a.length === 0) return;
			let o = Infinity, s = 0;
			for (let e of a) {
				let t = Number(e.offset), r = t + n(e);
				t < o && (o = t), r > s && (s = r);
			}
			let l = i + o, v = s - o, y = performance.now(), C = await c(l, v), w = performance.now() - y, T = performance.now(), E = {};
			for (let e of B) {
				let n = a.find((n) => n.name === t + e + ".weight");
				n && (E[e] = u(C, Number(n.offset) - o, n, !0).buf);
			}
			for (let e of z) {
				let n = a.find((n) => n.name === t + e + ".weight");
				if (n) {
					let t = Number(n.offset) - o;
					if (this.weightQuant === "q8") {
						let { quants: r, scales: i } = d(C, t, n);
						E[e] = r, E[`${e}__s`] = i;
					} else if (this.weightQuant === "q4k") {
						let { quants: r, meta: i } = f(C, t, n);
						E[e] = r, E[`${e}__s`] = i;
					} else E[e] = u(C, t, n, !0).buf;
				}
			}
			this.modelBuffers.layers[e] = E;
			let D = performance.now() - T;
			h += w, g += D, _ += v, x[e] = {
				layerIdx: e,
				fetchMs: w,
				processMs: D,
				bytes: v
			}, p += v, S++, this.reportProgress(p, m, `Layers ${S}/${b}`);
		}, w = /* @__PURE__ */ new Set();
		for (let e = 0; e < b; e++) {
			let t = C(e).finally(() => w.delete(t));
			w.add(t), w.size >= 4 && await Promise.race(w);
		}
		await Promise.all(w), this._loadTimings && (this._loadTimings.phases.layersFetch_ms = h, this._loadTimings.phases.layersProcess_ms = g, this._loadTimings.phases.layersTotal_ms = h + g, this._loadTimings.perLayer = x.filter((e) => e !== null), this._loadTimings.layerBytesTotal = _);
	}
	createWorkBuffers() {
		let e = this.config, t = e.hidden_size, n = e.q_dim, r = e.kv_dim, i = e.intermediate_size, a = e.context_length, o = e.vocab_size, s = e.num_kv_heads, c = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, l = c | GPUBufferUsage.COPY_DST;
		this.workBuffers = {
			hidden: this.device.createBuffer({
				size: t * 4,
				usage: l
			}),
			hiddenReadback: this.device.createBuffer({
				size: Math.max(t, e.num_layers * e.per_layer_input_dim, 1) * 4,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			}),
			residual: this.device.createBuffer({
				size: t * 4,
				usage: c
			}),
			normed: this.device.createBuffer({
				size: t * 4,
				usage: c
			}),
			q: this.device.createBuffer({
				size: n * 4,
				usage: c
			}),
			k: this.device.createBuffer({
				size: r * 4,
				usage: c
			}),
			v: this.device.createBuffer({
				size: r * 4,
				usage: c
			}),
			attnOut: this.device.createBuffer({
				size: n * 4,
				usage: c
			}),
			attnProj: this.device.createBuffer({
				size: t * 4,
				usage: c
			}),
			postAttnNormed: this.device.createBuffer({
				size: t * 4,
				usage: c
			}),
			attnScores: this.device.createBuffer({
				size: e.num_q_heads * a * 4,
				usage: c
			}),
			ffnGate: this.device.createBuffer({
				size: i * 4,
				usage: c
			}),
			ffnUp: this.device.createBuffer({
				size: i * 4,
				usage: c
			}),
			ffnMul: this.device.createBuffer({
				size: i * 4,
				usage: c
			}),
			ffnDown: this.device.createBuffer({
				size: t * 4,
				usage: c
			}),
			postFfnNormed: this.device.createBuffer({
				size: t * 4,
				usage: c
			}),
			logits: this.device.createBuffer({
				size: o * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			}),
			logitsReadback: this.device.createBuffer({
				size: o * 4,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			}),
			argmaxResult: this.device.createBuffer({
				size: 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			}),
			argmaxReadbackPool: [this.device.createBuffer({
				size: 4,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			}), this.device.createBuffer({
				size: 4,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			})],
			topk256Result: this.device.createBuffer({
				size: 256 * 2 * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
			}),
			topk256Readback: this.device.createBuffer({
				size: 256 * 2 * 4,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			}),
			plePmProjected: this.device.createBuffer({
				size: Math.max(4, e.num_layers * e.per_layer_input_dim * 4),
				usage: c
			}),
			pleInputs: this.device.createBuffer({
				size: Math.max(4, e.num_layers * e.per_layer_input_dim * 4),
				usage: c
			}),
			pleGate: this.device.createBuffer({
				size: Math.max(4, e.per_layer_input_dim * 4),
				usage: c
			}),
			pleGated: this.device.createBuffer({
				size: Math.max(4, e.per_layer_input_dim * 4),
				usage: c
			}),
			pleProjOut: this.device.createBuffer({
				size: t * 4,
				usage: c
			}),
			plePostNormed: this.device.createBuffer({
				size: t * 4,
				usage: c
			})
		}, this.kvCaches = [];
		for (let t = 0; t < e.num_layers; t++) {
			let n = N(t, e), r = a * s * n * 4;
			this.kvCaches.push({
				k: this.device.createBuffer({
					size: r,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
				}),
				v: this.device.createBuffer({
					size: r,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
				})
			});
		}
	}
	createBindGroups() {
		let e = this.config.arch === "qwen3", t = e ? void 0 : this.device.createBindGroup({
			layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: { buffer: this.workBuffers.hidden }
				},
				{
					binding: 1,
					resource: { buffer: this.modelBuffers.globals.per_layer_model_proj }
				},
				{
					binding: 2,
					resource: { buffer: this.workBuffers.plePmProjected }
				},
				{
					binding: 3,
					resource: { buffer: this.uniformBuffers.plePmProjMM }
				}
			]
		}), n = [];
		if (!e) for (let e = 0; e < this.config.num_layers; e++) n.push(this.device.createBindGroup({
			layout: this.pipelines.pleStage1Fuse.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: { buffer: this.workBuffers.plePmProjected }
				},
				{
					binding: 1,
					resource: { buffer: this.modelBuffers.globals.per_layer_proj_norm }
				},
				{
					binding: 2,
					resource: { buffer: this.modelBuffers.perLayerEmbeddings[e] }
				},
				{
					binding: 3,
					resource: { buffer: this.workBuffers.pleInputs }
				},
				{
					binding: 4,
					resource: { buffer: this.uniformBuffers.pleStage1[e] }
				}
			]
		}));
		let r = {
			embeddingLookup: this.device.createBindGroup({
				layout: this.pipelines.embeddingLookup.getBindGroupLayout(0),
				entries: [
					{
						binding: 0,
						resource: { buffer: this.modelBuffers.tokenEmbed }
					},
					{
						binding: 1,
						resource: { buffer: this.workBuffers.hidden }
					},
					{
						binding: 2,
						resource: { buffer: this.uniformBuffers.embeddingLookup }
					}
				]
			}),
			finalNorm: this.device.createBindGroup({
				layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
				entries: [
					{
						binding: 0,
						resource: { buffer: this.workBuffers.hidden }
					},
					{
						binding: 1,
						resource: { buffer: this.modelBuffers.finalNorm }
					},
					{
						binding: 2,
						resource: { buffer: this.workBuffers.normed }
					},
					{
						binding: 3,
						resource: { buffer: this.uniformBuffers.rmsNorm }
					}
				]
			}),
			lmHead: this.device.createBindGroup({
				layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
				entries: [
					{
						binding: 0,
						resource: { buffer: this.workBuffers.normed }
					},
					{
						binding: 1,
						resource: { buffer: this.modelBuffers.globals.lm_head ?? this.modelBuffers.tokenEmbed }
					},
					{
						binding: 2,
						resource: { buffer: this.workBuffers.logits }
					},
					{
						binding: 3,
						resource: { buffer: this.uniformBuffers.linearQ8_V_H }
					}
				]
			}),
			logitSoftcap: this.device.createBindGroup({
				layout: this.pipelines.logitSoftcap.getBindGroupLayout(0),
				entries: [{
					binding: 0,
					resource: { buffer: this.workBuffers.logits }
				}, {
					binding: 1,
					resource: { buffer: this.uniformBuffers.logitSoftcap }
				}]
			}),
			argmax: this.device.createBindGroup({
				layout: this.pipelines.argmax.getBindGroupLayout(0),
				entries: [
					{
						binding: 0,
						resource: { buffer: this.workBuffers.logits }
					},
					{
						binding: 1,
						resource: { buffer: this.workBuffers.argmaxResult }
					},
					{
						binding: 2,
						resource: { buffer: this.uniformBuffers.argmaxSize }
					}
				]
			}),
			topk256: this.device.createBindGroup({
				layout: this.pipelines.topk256.getBindGroupLayout(0),
				entries: [
					{
						binding: 0,
						resource: { buffer: this.workBuffers.logits }
					},
					{
						binding: 1,
						resource: { buffer: this.workBuffers.topk256Result }
					},
					{
						binding: 2,
						resource: { buffer: this.uniformBuffers.argmaxSize }
					}
				]
			}),
			layers: [],
			plePmProjMatmul: t,
			pleStage1Fuse: n
		};
		for (let e = 0; e < this.config.num_layers; e++) {
			let t = this.modelBuffers.layers[e], n = this.config.kv_producer_for_layer[e], i = this.kvCaches[n], a = this.kvCaches[e], o = {
				attnNorm: this.device.createBindGroup({
					layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.hidden }
						},
						{
							binding: 1,
							resource: { buffer: t.attn_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.normed }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.rmsNorm }
						}
					]
				}),
				linearQ: this.weightQuant === "f16" ? this.device.createBindGroup({
					layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.normed }
						},
						{
							binding: 1,
							resource: { buffer: t.attn_q }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.q }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.linearQ8_Q_H[e] }
						}
					]
				}) : this.mmQuant(this.workBuffers.normed, t, "attn_q", this.workBuffers.q, this.uniformBuffers.linearQ8_Q_H[e]),
				linearK: this.weightQuant === "f16" ? this.device.createBindGroup({
					layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.normed }
						},
						{
							binding: 1,
							resource: { buffer: t.attn_k }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.k }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.linearQ8_KV_H[e] }
						}
					]
				}) : this.mmQuant(this.workBuffers.normed, t, "attn_k", this.workBuffers.k, this.uniformBuffers.linearQ8_KV_H[e]),
				linearV: this.weightQuant === "f16" ? this.device.createBindGroup({
					layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.normed }
						},
						{
							binding: 1,
							resource: { buffer: t.attn_v }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.v }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.linearQ8_KV_H[e] }
						}
					]
				}) : this.mmQuant(this.workBuffers.normed, t, "attn_v", this.workBuffers.v, this.uniformBuffers.linearQ8_KV_H[e]),
				ropeQ: this.device.createBindGroup({
					layout: this.pipelines.rope.getBindGroupLayout(0),
					entries: [{
						binding: 0,
						resource: { buffer: this.workBuffers.q }
					}, {
						binding: 1,
						resource: { buffer: this.uniformBuffers.ropeQ[e] }
					}]
				}),
				ropeK: this.device.createBindGroup({
					layout: this.pipelines.rope.getBindGroupLayout(0),
					entries: [{
						binding: 0,
						resource: { buffer: this.workBuffers.k }
					}, {
						binding: 1,
						resource: { buffer: this.uniformBuffers.ropeK[e] }
					}]
				}),
				qNorm: this.device.createBindGroup({
					layout: this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.q }
						},
						{
							binding: 1,
							resource: { buffer: t.attn_q_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.uniformBuffers.perHeadRmsNormQ[e] }
						}
					]
				}),
				kNorm: this.device.createBindGroup({
					layout: this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.k }
						},
						{
							binding: 1,
							resource: { buffer: t.attn_k_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.uniformBuffers.perHeadRmsNormK[e] }
						}
					]
				}),
				vNorm: this.device.createBindGroup({
					layout: this.pipelines.perHeadRmsNormNoWeight.getBindGroupLayout(0),
					entries: [{
						binding: 0,
						resource: { buffer: this.workBuffers.v }
					}, {
						binding: 1,
						resource: { buffer: this.uniformBuffers.perHeadRmsNormV[e] }
					}]
				}),
				fusedNormRopeQ: this.device.createBindGroup({
					layout: this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.q }
						},
						{
							binding: 1,
							resource: { buffer: t.attn_q_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.uniformBuffers.fusedNormRopeQ[e] }
						},
						{
							binding: 3,
							resource: { buffer: this.modelBuffers.globals.rope_freqs ?? t.attn_q_norm }
						}
					]
				}),
				fusedNormRopeK: this.device.createBindGroup({
					layout: this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.k }
						},
						{
							binding: 1,
							resource: { buffer: t.attn_k_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.uniformBuffers.fusedNormRopeK[e] }
						},
						{
							binding: 3,
							resource: { buffer: this.modelBuffers.globals.rope_freqs ?? t.attn_k_norm }
						}
					]
				}),
				kvStore: this.device.createBindGroup({
					layout: this.pipelines.kvCacheStore.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.k }
						},
						{
							binding: 1,
							resource: { buffer: this.workBuffers.v }
						},
						{
							binding: 2,
							resource: { buffer: a.k }
						},
						{
							binding: 3,
							resource: { buffer: a.v }
						},
						{
							binding: 4,
							resource: { buffer: this.uniformBuffers.kvCacheStore[e] }
						}
					]
				}),
				attnScore: this.device.createBindGroup({
					layout: this.pipelines.attnScore.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.q }
						},
						{
							binding: 1,
							resource: { buffer: i.k }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.attnScores }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.attnScore[e] }
						}
					]
				}),
				softmax: this.device.createBindGroup({
					layout: this.pipelines.softmax.getBindGroupLayout(0),
					entries: [{
						binding: 0,
						resource: { buffer: this.workBuffers.attnScores }
					}, {
						binding: 1,
						resource: { buffer: this.uniformBuffers.softmax }
					}]
				}),
				attnOutput: this.device.createBindGroup({
					layout: this.pipelines.attnOutput.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.attnScores }
						},
						{
							binding: 1,
							resource: { buffer: i.v }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.attnOut }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.attnOutput[e] }
						}
					]
				}),
				linearAttnOut: this.weightQuant === "f16" ? this.device.createBindGroup({
					layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.attnOut }
						},
						{
							binding: 1,
							resource: { buffer: t.attn_output }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.attnProj }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.linearQ8_H_Q[e] }
						}
					]
				}) : this.mmQuant(this.workBuffers.attnOut, t, "attn_output", this.workBuffers.attnProj, this.uniformBuffers.linearQ8_H_Q[e]),
				postAttnNorm: this.device.createBindGroup({
					layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.attnProj }
						},
						{
							binding: 1,
							resource: { buffer: t.post_attention_norm ?? t.attn_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.postAttnNormed }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.rmsNorm }
						}
					]
				}),
				residualAdd1: this.device.createBindGroup({
					layout: this.pipelines.add.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.hidden }
						},
						{
							binding: 1,
							resource: { buffer: this.workBuffers.postAttnNormed }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.residual }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.sizeH }
						}
					]
				}),
				ffnNorm: this.device.createBindGroup({
					layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.residual }
						},
						{
							binding: 1,
							resource: { buffer: t.ffn_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.normed }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.rmsNorm }
						}
					]
				}),
				ffnGate: this.weightQuant === "f16" ? this.device.createBindGroup({
					layout: (this.mr4ForFfn ? this.pipelines.matmulQuantMR4 : this.pipelines.matmulQuant).getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.normed }
						},
						{
							binding: 1,
							resource: { buffer: t.ffn_gate }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.ffnGate }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.linearQ8_I_H[e] }
						}
					]
				}) : this.mmQuant(this.workBuffers.normed, t, "ffn_gate", this.workBuffers.ffnGate, this.uniformBuffers.linearQ8_I_H[e]),
				ffnUp: this.weightQuant === "f16" ? this.device.createBindGroup({
					layout: (this.mr4ForFfn ? this.pipelines.matmulQuantMR4 : this.pipelines.matmulQuant).getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.normed }
						},
						{
							binding: 1,
							resource: { buffer: t.ffn_up }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.ffnUp }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.linearQ8_I_H[e] }
						}
					]
				}) : this.mmQuant(this.workBuffers.normed, t, "ffn_up", this.workBuffers.ffnUp, this.uniformBuffers.linearQ8_I_H[e]),
				geluMul: this.device.createBindGroup({
					layout: (this.config.ffn_activation === "silu" ? this.pipelines.siluMul : this.pipelines.geluMul).getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.ffnGate }
						},
						{
							binding: 1,
							resource: { buffer: this.workBuffers.ffnUp }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.ffnMul }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.sizeI[e] }
						}
					]
				}),
				ffnDown: this.weightQuant === "f16" ? this.device.createBindGroup({
					layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.ffnMul }
						},
						{
							binding: 1,
							resource: { buffer: t.ffn_down }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.ffnDown }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.linearQ8_H_I[e] }
						}
					]
				}) : this.mmQuant(this.workBuffers.ffnMul, t, "ffn_down", this.workBuffers.ffnDown, this.uniformBuffers.linearQ8_H_I[e]),
				postFfnNorm: this.device.createBindGroup({
					layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.ffnDown }
						},
						{
							binding: 1,
							resource: { buffer: t.post_ffw_norm ?? t.ffn_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.postFfnNormed }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.rmsNorm }
						}
					]
				}),
				residualAdd2: this.device.createBindGroup({
					layout: this.pipelines.add.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.residual }
						},
						{
							binding: 1,
							resource: { buffer: this.workBuffers.postFfnNormed }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.hidden }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.sizeH }
						}
					]
				}),
				fusedPostAttnNormAdd: this.device.createBindGroup({
					layout: this.pipelines.fusedNormAdd.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.attnProj }
						},
						{
							binding: 1,
							resource: { buffer: t.post_attention_norm ?? t.attn_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.hidden }
						},
						{
							binding: 3,
							resource: { buffer: this.workBuffers.residual }
						},
						{
							binding: 4,
							resource: { buffer: this.uniformBuffers.rmsNorm }
						}
					]
				}),
				fusedPostFfnNormAdd: this.device.createBindGroup({
					layout: this.pipelines.fusedNormAdd.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.ffnDown }
						},
						{
							binding: 1,
							resource: { buffer: t.post_ffw_norm ?? t.ffn_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.residual }
						},
						{
							binding: 3,
							resource: { buffer: this.workBuffers.hidden }
						},
						{
							binding: 4,
							resource: { buffer: this.uniformBuffers.rmsNorm }
						}
					]
				}),
				pleInpGateMatmul: this.device.createBindGroup({
					layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.hidden }
						},
						{
							binding: 1,
							resource: { buffer: t.inp_gate ?? t.attn_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.pleGate }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.pleInpGateMM }
						}
					]
				}),
				pleGeluMul: this.device.createBindGroup({
					layout: this.pipelines.pleGeluMul.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.pleGate }
						},
						{
							binding: 1,
							resource: { buffer: this.workBuffers.pleInputs }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.pleGated }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.pleGeluMulParams[e] }
						}
					]
				}),
				plePostProjMatmul: this.device.createBindGroup({
					layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.pleGated }
						},
						{
							binding: 1,
							resource: { buffer: t.proj ?? t.attn_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.pleProjOut }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.plePostProjMM }
						}
					]
				}),
				plePostNorm: this.device.createBindGroup({
					layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.pleProjOut }
						},
						{
							binding: 1,
							resource: { buffer: t.post_norm ?? t.attn_norm }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.plePostNormed }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.rmsNorm }
						}
					]
				}),
				pleSkipScaleAdd: this.device.createBindGroup({
					layout: this.pipelines.pleSkipScaleAdd.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.hidden }
						},
						{
							binding: 1,
							resource: { buffer: this.workBuffers.plePostNormed }
						},
						{
							binding: 2,
							resource: { buffer: t.layer_output_scale ?? t.attn_norm }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.pleSkipScaleAdd }
						}
					]
				}),
				qwenAttnAdd: this.device.createBindGroup({
					layout: this.pipelines.add.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.hidden }
						},
						{
							binding: 1,
							resource: { buffer: this.workBuffers.attnProj }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.residual }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.sizeH }
						}
					]
				}),
				qwenFfnAdd: this.device.createBindGroup({
					layout: this.pipelines.add.getBindGroupLayout(0),
					entries: [
						{
							binding: 0,
							resource: { buffer: this.workBuffers.residual }
						},
						{
							binding: 1,
							resource: { buffer: this.workBuffers.ffnDown }
						},
						{
							binding: 2,
							resource: { buffer: this.workBuffers.hidden }
						},
						{
							binding: 3,
							resource: { buffer: this.uniformBuffers.sizeH }
						}
					]
				})
			};
			r.layers.push(o);
		}
		this.bindGroupCache = r;
	}
	encodeTransformerPass(e, t, n, r) {
		let i = this.config, a = i.hidden_size, o = i.num_q_heads, s = i.num_kv_heads, c = n + 1;
		this.profileSlotCursor = 0, this.profilePassLabels = [];
		let l = this.cpuProfileActive && this.cpuProfileCapturing, u = l ? performance.now() : 0;
		t !== null && this.wb(this.uniformBuffers.embeddingLookup, 4, new Uint32Array([t]));
		let d = new Uint32Array([n]), f = new Uint32Array([c]), p = t === null ? null : new Uint32Array([t]);
		for (let e = 0; e < i.num_layers; e++) this.wb(this.uniformBuffers.fusedNormRopeQ[e], 12, d), this.wb(this.uniformBuffers.fusedNormRopeK[e], 12, d), this.wb(this.uniformBuffers.kvCacheStore[e], 8, d), this.wb(this.uniformBuffers.attnScore[e], 12, f), this.wb(this.uniformBuffers.attnOutput[e], 12, f), p !== null && this.wb(this.uniformBuffers.pleStage1[e], 4, p);
		this.wb(this.uniformBuffers.softmax, 4, f);
		let m = l ? performance.now() : 0, h;
		if (h = this.beginPass(e, "embed"), h.setPipeline(this.pipelines.embeddingLookup), h.setBindGroup(0, this.bindGroupCache.embeddingLookup), h.dispatchWorkgroups(Math.ceil(a / 256)), h.end(), r?.kind !== "embed") {
			if (i.arch !== "qwen3") {
				let t = i.num_layers * i.per_layer_input_dim;
				if (h = this.beginPass(e, "ple1.pmProj"), h.setPipeline(this.pipelines.matmulQuant), h.setBindGroup(0, this.bindGroupCache.plePmProjMatmul), this.dispatchMatmul(h, t), h.end(), r?.kind === "plePmProjected") return;
				h = this.beginPass(e, "ple1.stage1Fuse"), h.setPipeline(this.pipelines.pleStage1Fuse);
				for (let e = 0; e < i.num_layers; e++) h.setBindGroup(0, this.bindGroupCache.pleStage1Fuse[e]), h.dispatchWorkgroups(1);
				if (h.end(), r?.kind === "pleStage1") return;
			}
			for (let t = 0; t < i.num_layers; t++) {
				let n = this.bindGroupCache.layers[t], l = N(t, i), u = P(t, i), d = o * l, f = s * l, p = F(t, i), m = this.weightQuant === "q8" ? this.pipelines.matmulQ8Mr : this.weightQuant === "q4k" ? this.pipelines.matmulQ4KMr : this.mr4ForFfn ? this.pipelines.matmulQuantMR4 : this.pipelines.matmulQuant;
				if (h = this.beginPass(e, "attn.rmsNorm"), h.setPipeline(this.pipelines.rmsNorm), h.setBindGroup(0, n.attnNorm), h.dispatchWorkgroups(1), h.end(), h = this.beginPass(e, p ? "attn.linearQKV.producer" : "attn.linearQ.consumer"), h.setPipeline(m), h.setBindGroup(0, n.linearQ), this.dispatchMatmulRows(h, d), p && (h.setBindGroup(0, n.linearK), this.dispatchMatmulRows(h, f), h.setBindGroup(0, n.linearV), this.dispatchMatmulRows(h, f)), h.end(), r && "layer" in r && r.layer === t && (r.kind === "preRopeQ" || r.kind === "preRopeK" || r.kind === "preRopeV") || (p && i.v_norm !== !1 && (h = this.beginPass(e, "attn.vNorm"), h.setPipeline(this.pipelines.perHeadRmsNormNoWeight), h.setBindGroup(0, n.vNorm), h.dispatchWorkgroups(s), h.end()), h = this.beginPass(e, p ? "attn.fusedNormRopeQK" : "attn.fusedNormRopeQ"), h.setPipeline(this.pipelines.fusedPerHeadNormRope), h.setBindGroup(0, n.fusedNormRopeQ), h.dispatchWorkgroups(o), p && (h.setBindGroup(0, n.fusedNormRopeK), h.dispatchWorkgroups(s)), h.end(), r && "layer" in r && r.layer === t && (r.kind === "postRopeQ" || r.kind === "postRopeK")) || (p && (h = this.beginPass(e, "attn.kvCacheStore"), h.setPipeline(this.pipelines.kvCacheStore), h.setBindGroup(0, n.kvStore), h.dispatchWorkgroups(Math.ceil(f / 256)), h.end()), h = this.beginPass(e, "attn.attnScore"), h.setPipeline(this.pipelines.attnScore), h.setBindGroup(0, n.attnScore), h.dispatchWorkgroups(Math.ceil(o * c / 256)), h.end(), h = this.beginPass(e, "attn.softmax"), h.setPipeline(this.pipelines.softmax), h.setBindGroup(0, n.softmax), h.dispatchWorkgroups(o), h.end(), h = this.beginPass(e, "attn.attnOutput"), h.setPipeline(this.pipelines.attnOutput), h.setBindGroup(0, n.attnOutput), h.dispatchWorkgroups(Math.ceil(o * l / 256)), h.end(), r && r.kind === "attnOut" && r.layer === t)) return;
				h = this.beginPass(e, "attn.linearOut"), h.setPipeline(m), h.setBindGroup(0, n.linearAttnOut), this.dispatchMatmulRows(h, a), h.end(), i.post_attn_norm === !1 ? (h = this.beginPass(e, "attn.residualAdd"), h.setPipeline(this.pipelines.add), h.setBindGroup(0, n.qwenAttnAdd), h.dispatchWorkgroups(Math.ceil(a / 256)), h.end()) : (h = this.beginPass(e, "attn.postNormAdd"), h.setPipeline(this.pipelines.fusedNormAdd), h.setBindGroup(0, n.fusedPostAttnNormAdd), h.dispatchWorkgroups(1), h.end()), h = this.beginPass(e, "ffn.rmsNorm"), h.setPipeline(this.pipelines.rmsNorm), h.setBindGroup(0, n.ffnNorm), h.dispatchWorkgroups(1), h.end(), h = this.beginPass(e, "ffn.linearGateUp"), h.setPipeline(m), h.setBindGroup(0, n.ffnGate), this.dispatchMatmulRows(h, u), h.setBindGroup(0, n.ffnUp), this.dispatchMatmulRows(h, u), h.end();
				let g = i.ffn_activation === "silu" ? this.pipelines.siluMul : this.pipelines.geluMul;
				if (h = this.beginPass(e, "ffn.actMul"), h.setPipeline(g), h.setBindGroup(0, n.geluMul), h.dispatchWorkgroups(Math.ceil(u / 256)), h.end(), h = this.beginPass(e, "ffn.linearDown"), h.setPipeline(m), h.setBindGroup(0, n.ffnDown), this.dispatchMatmulRows(h, a), h.end(), i.post_ffn_norm === !1 ? (h = this.beginPass(e, "ffn.residualAdd"), h.setPipeline(this.pipelines.add), h.setBindGroup(0, n.qwenFfnAdd), h.dispatchWorkgroups(Math.ceil(a / 256)), h.end()) : (h = this.beginPass(e, "ffn.postNormAdd"), h.setPipeline(this.pipelines.fusedNormAdd), h.setBindGroup(0, n.fusedPostFfnNormAdd), h.dispatchWorkgroups(1), h.end()), i.arch !== "qwen3") {
					let t = i.per_layer_input_dim;
					h = this.beginPass(e, "ple2.linearInpGate"), h.setPipeline(this.pipelines.matmulQuant), h.setBindGroup(0, n.pleInpGateMatmul), this.dispatchMatmul(h, t), h.end(), h = this.beginPass(e, "ple2.geluMul"), h.setPipeline(this.pipelines.pleGeluMul), h.setBindGroup(0, n.pleGeluMul), h.dispatchWorkgroups(Math.ceil(t / 256)), h.end(), h = this.beginPass(e, "ple2.linearPostProj"), h.setPipeline(this.pipelines.matmulQuant), h.setBindGroup(0, n.plePostProjMatmul), this.dispatchMatmul(h, a), h.end(), h = this.beginPass(e, "ple2.rmsNorm"), h.setPipeline(this.pipelines.rmsNorm), h.setBindGroup(0, n.plePostNorm), h.dispatchWorkgroups(1), h.end(), h = this.beginPass(e, "ple2.skipScaleAdd"), h.setPipeline(this.pipelines.pleSkipScaleAdd), h.setBindGroup(0, n.pleSkipScaleAdd), h.dispatchWorkgroups(Math.ceil(a / 256)), h.end();
				}
				if (r?.kind === "afterLayer" && r.layer === t) return;
			}
			if (h = this.beginPass(e, "final.rmsNorm"), h.setPipeline(this.pipelines.rmsNorm), h.setBindGroup(0, this.bindGroupCache.finalNorm), h.dispatchWorkgroups(1), h.end(), r?.kind === "logits") {
				let t = this.config.vocab_size;
				h = this.beginPass(e, "lmHead"), h.setPipeline(this.pipelines.matmulQuant), h.setBindGroup(0, this.bindGroupCache.lmHead), this.dispatchMatmul(h, t), h.end(), this.config.final_logit_softcapping > 0 && (h = this.beginPass(e, "logitSoftcap"), h.setPipeline(this.pipelines.logitSoftcap), h.setBindGroup(0, this.bindGroupCache.logitSoftcap), h.dispatchWorkgroups(Math.ceil(t / 256)), h.end());
			}
			if (l) {
				let e = performance.now();
				this.recordCpuPhase("cpu.encode.uniforms", m - u), this.recordCpuPhase("cpu.encode.transformerPasses", e - m);
			}
		}
	}
	sampleNextTokenSubmit(e, t, n, r, i) {
		if (this.deviceLost) throw Error("WebGPU device lost");
		let a = this.cpuProfileActive && this.cpuProfileCapturing, o = a ? performance.now() : 0, s = this.config.vocab_size, c = this.config, l;
		l = this.beginPass(e, "lmHead"), l.setPipeline(this.pipelines.matmulQuant), l.setBindGroup(0, this.bindGroupCache.lmHead), this.dispatchMatmul(l, s), l.end(), this.config.final_logit_softcapping > 0 && (l = this.beginPass(e, "logitSoftcap"), l.setPipeline(this.pipelines.logitSoftcap), l.setBindGroup(0, this.bindGroupCache.logitSoftcap), l.dispatchWorkgroups(Math.ceil(s / 256)), l.end());
		let u = this.tuning.pipeline.decodeDepth === 2 && this.tuning.pipeline.greedyFastPath && t === 0 && r <= 1, d = null;
		if (u) {
			l = this.beginPass(e, "sample.argmax"), l.setPipeline(this.pipelines.argmax), l.setBindGroup(0, this.bindGroupCache.argmax), l.dispatchWorkgroups(1), l.end(), e.copyBufferToBuffer(this.workBuffers.argmaxResult, 0, this.uniformBuffers.embeddingLookup, 4, 4);
			for (let t = 0; t < c.num_layers; t++) e.copyBufferToBuffer(this.workBuffers.argmaxResult, 0, this.uniformBuffers.pleStage1[t], 4, 4);
			d = this.workBuffers.argmaxReadbackPool[this.argmaxReadbackIdx], this.argmaxReadbackIdx ^= 1, e.copyBufferToBuffer(this.workBuffers.argmaxResult, 0, d, 0, 4);
		} else l = this.beginPass(e, "sample.topk256"), l.setPipeline(this.pipelines.topk256), l.setBindGroup(0, this.bindGroupCache.topk256), l.dispatchWorkgroups(1), l.end(), e.copyBufferToBuffer(this.workBuffers.topk256Result, 0, this.workBuffers.topk256Readback, 0, 256 * 2 * 4);
		let f = a ? performance.now() : 0, p = this.appendProfileResolve(e), m = a ? performance.now() : 0, h = e.finish(), g = a ? performance.now() : 0;
		this.device.queue.submit([h]);
		let _ = a ? performance.now() : 0, v = p ? this.collectProfileSamples() : null;
		return a && (this.recordCpuPhase("cpu.encode.samplePasses", f - o), this.recordCpuPhase("cpu.appendProfileResolve", m - f), this.recordCpuPhase("cpu.encoder.finish", g - m), this.recordCpuPhase("cpu.queue.submit", _ - g)), {
			readback: (async () => {
				let e = a ? performance.now() : 0, o;
				if (u) {
					let t = d;
					try {
						await t.mapAsync(GPUMapMode.READ);
					} catch (e) {
						throw Error(`GPU readback failed (device lost?): ${e}`);
					}
					let n = a ? performance.now() : 0;
					o = new Uint32Array(t.getMappedRange())[0], t.unmap();
					let r = a ? performance.now() : 0;
					a && (this.recordCpuPhase("cpu.mapAsync.wait", n - e), this.recordCpuPhase("cpu.readback.process", r - n));
				} else {
					try {
						await this.workBuffers.topk256Readback.mapAsync(GPUMapMode.READ);
					} catch (e) {
						throw Error(`GPU readback failed (device lost?): ${e}`);
					}
					let s = a ? performance.now() : 0, c = new Float32Array(this.workBuffers.topk256Readback.getMappedRange().slice(0));
					this.workBuffers.topk256Readback.unmap();
					let l = Array(256), u = new Uint32Array(c.buffer.slice(0));
					for (let e = 0; e < 256; e++) l[e] = {
						val: c[e * 2],
						id: u[e * 2 + 1]
					};
					if (r > 1 && i.length > 0) {
						let e = new Set(i);
						for (let t = 0; t < 256; t++) e.has(l[t].id) && (l[t].val > 0 ? l[t].val /= r : l[t].val *= r);
					}
					if (l.sort((e, t) => t.val - e.val), t === 0) o = l[0].id;
					else {
						let e = l[0].val, r = 0, i = new Float32Array(256);
						for (let n = 0; n < 256; n++) i[n] = Math.exp((l[n].val - e) / t), r += i[n];
						let a = 0, s = 256;
						for (let e = 0; e < 256; e++) if (a += i[e] / r, a >= n) {
							s = e + 1;
							break;
						}
						let c = 0;
						for (let e = 0; e < s; e++) c += i[e];
						let u = Math.random() * c, d = l[s - 1].id;
						for (let e = 0; e < s; e++) if (u -= i[e], u <= 0) {
							d = l[e].id;
							break;
						}
						o = d;
					}
					a && this.recordCpuPhase("cpu.mapAsync.wait", s - e);
				}
				return o;
			})(),
			profilePromise: v,
			useGreedyFast: u
		};
	}
	async sampleNextToken(e, t, n, r, i) {
		let a = this.sampleNextTokenSubmit(e, t, n, r, i), o = await a.readback;
		return a.profilePromise && await a.profilePromise, o;
	}
	async forwardPassAndGetToken(e, t, n = 0, r = .9, i = 1, a = []) {
		let o = this.cpuProfileActive && this.cpuProfileCapturing, s = o ? performance.now() : 0;
		o && (this.cpuWbCountThisForward = 0, this.cpuPassCountThisForward = 0, this.cpuPrevForwardEnd_ms > 0 && this.recordCpuPhase("cpu.betweenForwards", s - this.cpuPrevForwardEnd_ms));
		let c = this.device.createCommandEncoder();
		this.encodeTransformerPass(c, e, t);
		let l = await this.sampleNextToken(c, n, r, i, a);
		if (o) {
			let e = performance.now();
			this.cpuForwardSamples_ms.push(e - s), this.cpuForwardWriteBufferCounts.push(this.cpuWbCountThisForward), this.cpuForwardPassEncodeCounts.push(this.cpuPassCountThisForward), this.cpuPrevForwardEnd_ms = e;
		}
		return l;
	}
	forwardPassOnly(e, t) {
		let n = this.device.createCommandEncoder();
		this.encodeTransformerPass(n, e, t), this.device.queue.submit([n.finish()]);
	}
	forwardPassSubmitOnly(e, t, n) {
		let r = this.device.createCommandEncoder();
		this.encodeTransformerPass(r, e, t);
		let i = this.sampleNextTokenSubmit(r, 0, .9, 1, n);
		return {
			readback: i.readback,
			profilePromise: i.profilePromise
		};
	}
	async prefillBatched(e, t = 0) {
		let n = this.profileCapturing, r = this.cpuProfileCapturing;
		this.profileCapturing = !1, this.cpuProfileCapturing = !1;
		for (let n = 0; n < e.length; n++) this.forwardPassOnly(e[n], t + n);
		await this.device.queue.onSubmittedWorkDone(), this.profileCapturing = n, this.cpuProfileCapturing = r;
	}
	resetKVCaches() {
		let e = this.config.head_dim, t = this.config.num_kv_heads, n = this.config.context_length, r = new Float32Array(n * t * e);
		for (let e = 0; e < this.config.num_layers; e++) this.wb(this.kvCaches[e].k, 0, r), this.wb(this.kvCaches[e].v, 0, r);
	}
	async getFirstTokenAfterPrefill(e, t, n, r) {
		let i = this.device.createCommandEncoder();
		return this.sampleNextToken(i, e, t, n, r);
	}
	addUserMessage(e) {
		this.conversationHistory.push({
			role: "user",
			text: e
		});
	}
	async *generate(e = {}) {
		if (this.deviceLost) throw Error("WebGPU device lost — call dispose() and recreate the engine");
		let t = e.temperature ?? .7, n = e.topP ?? .9, r = e.repPenalty ?? 1.2, i = e.maxTokens ?? 32768, a = e.toolsJson ?? "[]", o = e.signal, s = performance.now(), c = 0, l = s, u = 0, d = s, f = 0, p = !1, m;
		if (this.kvPosition === 0) {
			let e = h(this.conversationHistory, a, this._systemPrompt);
			m = this.tokenizer.encode(e);
		} else {
			let e = `<end_of_turn>\n<start_of_turn>user\n${this.conversationHistory[this.conversationHistory.length - 1].text}<end_of_turn>\n<start_of_turn>model\n`;
			m = this.tokenizer.encode(e).slice(1);
		}
		if (this.kvPosition + m.length >= this.config.context_length - 10) {
			this.conversationHistory = [{
				role: "user",
				text: this.conversationHistory[this.conversationHistory.length - 1].text
			}];
			let e = h(this.conversationHistory, a, this._systemPrompt);
			m = this.tokenizer.encode(e), this.resetKVCaches(), this.kvPosition = 0;
		}
		await this.prefillBatched(m, this.kvPosition);
		let g = [...m];
		this.kvPosition += m.length;
		let _ = await this.getFirstTokenAfterPrefill(t, n, r, g);
		g.push(_);
		let v = this.tokenizer.funcTokens["<end_function_call>"], y = [_];
		f = performance.now() - s, u = 1, d = performance.now(), yield this.tokenizer.decodeToken(_), l = performance.now();
		let b = this.tuning.pipeline.decodeDepth === 2 && this.tuning.pipeline.greedyFastPath && t === 0 && r <= 1, x = (e) => e === 1 || e === 0 || e === 50 || e === 106, S = 0;
		if (b) {
			let e = null, t = 0;
			if (!(x(_) || v && _ === v) && i > 1 && !o?.aborted) {
				let n = this.kvPosition + t;
				n < this.config.context_length - 1 && (e = this.forwardPassSubmitOnly(_, n, g), t++);
			}
			for (let n = 1; n < i && e !== null; n++) {
				if (o?.aborted) {
					await e.readback, e.profilePromise && await e.profilePromise, e = null;
					break;
				}
				let r = null, a = this.kvPosition + t;
				if (n + 1 < i && a < this.config.context_length - 1 && (r = this.forwardPassSubmitOnly(null, a, g), t++), _ = await e.readback, e.profilePromise && await e.profilePromise, e = r, x(_)) {
					S++, e &&= (await e.readback, e.profilePromise && await e.profilePromise, null);
					break;
				}
				if (v && _ === v) {
					S++, g.push(_), y.push(_), e &&= (await e.readback, e.profilePromise && await e.profilePromise, null);
					break;
				}
				S++, g.push(_), y.push(_), c += performance.now() - l, u++, d = performance.now(), yield this.tokenizer.decodeToken(_), l = performance.now();
			}
		} else for (let e = 1; e < i && !(x(_) || v && _ === v || o?.aborted); e++) {
			let e = this.kvPosition + S;
			if (e >= this.config.context_length - 1 || (_ = await this.forwardPassAndGetToken(_, e, t, n, r, g), S++, x(_))) break;
			if (v && _ === v) {
				g.push(_), y.push(_);
				break;
			}
			g.push(_), y.push(_), c += performance.now() - l, u++, d = performance.now(), yield this.tokenizer.decodeToken(_), l = performance.now();
		}
		p = o?.aborted === !0;
		let C = this.tokenizer.decodeTokens(y);
		this.conversationHistory.push({
			role: "model",
			text: C
		}), this.kvPosition += S, this._lastGenerateStats = {
			tokens: u,
			firstTokenMs: f,
			coreDecodeMs: c,
			totalWallMs: d - s,
			aborted: p
		};
	}
	resetConversation() {
		this.conversationHistory = [], this.kvPosition = 0, this.resetKVCaches();
	}
	get systemPrompt() {
		return this._systemPrompt;
	}
	get lastGenerateStats() {
		return this._lastGenerateStats;
	}
	setSystemPrompt(e) {
		let t = e?.trim() ?? "";
		this._systemPrompt = t === "" ? null : t;
	}
	async captureHidden(e, t, n) {
		if (this.deviceLost) throw Error("WebGPU device lost");
		let r = this.device.createCommandEncoder();
		this.encodeTransformerPass(r, e, t, n);
		let i, a, o = this.workBuffers.hiddenReadback;
		switch (n.kind) {
			case "embed":
			case "afterLayer":
				i = this.workBuffers.hidden, a = this.config.hidden_size;
				break;
			case "plePmProjected":
				i = this.workBuffers.plePmProjected, a = this.config.num_layers * this.config.per_layer_input_dim;
				break;
			case "pleStage1":
				i = this.workBuffers.pleInputs, a = this.config.num_layers * this.config.per_layer_input_dim;
				break;
			case "final":
				i = this.workBuffers.normed, a = this.config.hidden_size;
				break;
			case "logits":
				i = this.workBuffers.logits, a = this.config.vocab_size, o = this.workBuffers.logitsReadback;
				break;
			case "preRopeQ":
			case "postRopeQ":
			case "attnOut": {
				let e = N(n.layer, this.config);
				i = n.kind === "attnOut" ? this.workBuffers.attnOut : this.workBuffers.q, a = this.config.num_q_heads * e;
				break;
			}
			case "preRopeK":
			case "postRopeK": {
				let e = N(n.layer, this.config);
				i = this.workBuffers.k, a = this.config.num_kv_heads * e;
				break;
			}
			case "preRopeV": {
				let e = N(n.layer, this.config);
				i = this.workBuffers.v, a = this.config.num_kv_heads * e;
				break;
			}
		}
		let s = a * 4;
		r.copyBufferToBuffer(i, 0, o, 0, s), this.device.queue.submit([r.finish()]), await o.mapAsync(GPUMapMode.READ, 0, s);
		let c = o.getMappedRange(0, s), l = new Float32Array(c.slice(0));
		return o.unmap(), l;
	}
	encodePromptTokens(e, t) {
		if (t) {
			let t = h([{
				role: "user",
				text: e
			}], "[]");
			return this.tokenizer.encode(t);
		}
		return this.tokenizer.encode(e);
	}
	async prefillForCapture(e, t) {
		e.length !== 0 && (await this.prefillBatched(e, t), this.kvPosition = t + e.length);
	}
	resetKVForCapture() {
		this.resetKVCaches(), this.kvPosition = 0;
	}
	async captureEmbedSliceAll(e) {
		if (this.deviceLost) throw Error("WebGPU device lost");
		let t = this.config.per_layer_input_dim;
		if (t === 0) throw Error("captureEmbedSliceAll: no PLE on this model");
		let n = this.config.num_layers, r = t * 2, i = n * r;
		if (i > this.workBuffers.hiddenReadback.size) throw Error(`captureEmbedSliceAll: readback buffer too small (${i} > ${this.workBuffers.hiddenReadback.size})`);
		let a = this.device.createCommandEncoder(), o = e * r;
		for (let e = 0; e < n; e++) a.copyBufferToBuffer(this.modelBuffers.perLayerEmbeddings[e], o, this.workBuffers.hiddenReadback, e * r, r);
		this.device.queue.submit([a.finish()]), await this.workBuffers.hiddenReadback.mapAsync(GPUMapMode.READ, 0, i);
		let s = this.workBuffers.hiddenReadback.getMappedRange(0, i).slice(0);
		this.workBuffers.hiddenReadback.unmap();
		let c = new R(s), l = [];
		for (let e = 0; e < n; e++) {
			let n = new Float32Array(t);
			for (let r = 0; r < t; r++) n[r] = c[e * t + r];
			l.push(n);
		}
		return l;
	}
	getGgufTensorInfo(e) {
		if (!this.ggufTensors) return null;
		let t = this.ggufTensors.find((t) => t.name === e + ".weight" || t.name === e);
		return t ? {
			dims: t.dims.map((e) => Number(e)),
			type: t.type,
			offset: Number(t.offset),
			byteSize: n(t)
		} : null;
	}
	async readGlobalTensor(e, t, n) {
		let r = this.modelBuffers.globals[e];
		if (!r) throw Error(`readGlobalTensor: no global tensor '${e}' uploaded`);
		return this.readF16Buffer(r, t, n, `global:${e}`);
	}
	async readLayerTensor(e, t, n, r) {
		let i = this.modelBuffers.layers;
		if (e < 0 || e >= i.length) throw Error(`readLayerTensor: layer ${e} out of range [0, ${i.length})`);
		let a = i[e][t];
		if (!a) throw Error(`readLayerTensor: no per-layer tensor '${t}' at layer ${e}`);
		return this.readF16Buffer(a, n, r, `layer${e}:${t}`);
	}
	async readEmbeddingTensor(e, t) {
		let n = this.modelBuffers.tokenEmbed;
		if (!n) throw Error("readEmbeddingTensor: embedding buffer not uploaded");
		return this.readF16Buffer(n, e, t, "tokenEmbed");
	}
	async readF16Buffer(e, t, n, r) {
		if (this.deviceLost) throw Error("WebGPU device lost");
		let i = t * 2, a = n * 2;
		if (i + a > e.size) throw Error(`${r}: slice [${t}..${t + n}) out of range (${e.size / 2} elems)`);
		let o = i & -4, s = (i - o) / 2, c = a + (i - o) + 3 & -4, l = this.device.createBuffer({
			size: c,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
		}), u = this.device.createCommandEncoder();
		u.copyBufferToBuffer(e, o, l, 0, c), this.device.queue.submit([u.finish()]), await l.mapAsync(GPUMapMode.READ);
		let d = l.getMappedRange().slice(0);
		l.unmap(), l.destroy();
		let f = new R(d, 0, s + n), p = new Float32Array(n);
		for (let e = 0; e < n; e++) p[e] = f[s + e];
		return p;
	}
	async fetchRawTensorSlice(e, n, r) {
		if (!this.ggufTensors || this.ggufDataOffset === void 0) throw Error("fetchRawTensorSlice: GGUF metadata not persisted");
		if (!this.modelFile) throw Error("fetchRawTensorSlice: Range refetch unavailable (loaded from buffer, not URL)");
		let i = this.ggufTensors.find((t) => t.name === e + ".weight" || t.name === e);
		if (!i) throw Error(`fetchRawTensorSlice: no tensor named ${e}`);
		let a = Number(i.dims.reduce((e, t) => e * t, 1n));
		if (n < 0 || n + r > a) throw Error(`fetchRawTensorSlice: [${n}..${n + r}) exceeds tensor length ${a}`);
		if (i.type === 0 || i.type === 1 || i.type === 30) {
			let e = i.type === 0 ? 4 : 2, t = this.ggufDataOffset + Number(i.offset) + n * e, a = r * e, o = await fetch(this.modelFile, { headers: { Range: `bytes=${t}-${t + a - 1}` } });
			if (!o.ok && o.status !== 206) throw Error(`fetchRawTensorSlice: HTTP ${o.status} for range ${t}+${a}`);
			let s = new Uint8Array(await o.arrayBuffer());
			if (i.type === 30) {
				let e = new Float32Array(r), t = /* @__PURE__ */ new ArrayBuffer(4), n = new Uint32Array(t), i = new Float32Array(t), a = new DataView(s.buffer, s.byteOffset, s.byteLength);
				for (let t = 0; t < r; t++) n[0] = a.getUint16(t * 2, !0) << 16, e[t] = i[0];
				return e;
			}
			if (i.type === 1) {
				let e = new R(s.buffer, s.byteOffset, r), t = new Float32Array(r);
				for (let n = 0; n < r; n++) t[n] = e[n];
				return t;
			}
			return new Float32Array(s.buffer.slice(s.byteOffset, s.byteOffset + a));
		}
		let o = {
			8: {
				elems: 32,
				bytes: 34,
				dequant: (e, t, n) => e.dequantizeQ8_0(t, n)
			},
			12: {
				elems: 256,
				bytes: 144,
				dequant: (e, t, n) => e.dequantizeQ4_K(t, n)
			},
			13: {
				elems: 256,
				bytes: 176,
				dequant: (e, t, n) => e.dequantizeQ5_K(t, n)
			},
			14: {
				elems: 256,
				bytes: 210,
				dequant: (e, t, n) => e.dequantizeQ6_K(t, n)
			}
		}[i.type];
		if (!o) throw Error(`fetchRawTensorSlice: unsupported source type ${i.type}`);
		let s = Math.floor(n / o.elems), c = Math.ceil((n + r) / o.elems) - s, l = this.ggufDataOffset + Number(i.offset) + s * o.bytes, u = c * o.bytes, d = await fetch(this.modelFile, { headers: { Range: `bytes=${l}-${l + u - 1}` } });
		if (!d.ok && d.status !== 206) throw Error(`fetchRawTensorSlice: HTTP ${d.status} for quant-range ${l}+${u}`);
		let f = new t(new Uint8Array(await d.arrayBuffer())), p = c * o.elems, m = o.dequant(f, 0, p), h = n - s * o.elems;
		return m.subarray(h, h + r).slice();
	}
	getProfileCapability() {
		return { ...this.profileCapability };
	}
	enableProfile() {
		if (!this.profileCapability.timestampQuerySupported) {
			this.profileActive = !1, this.profileCapturing = !1;
			return;
		}
		if (!this.profileQuerySet) {
			let e = this.profileCapability.querySlots;
			this.profileQuerySet = this.device.createQuerySet({
				type: "timestamp",
				count: e
			}), this.profileResolveBuf = this.device.createBuffer({
				size: e * 8,
				usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
			}), this.profileStagingBuf = this.device.createBuffer({
				size: e * 8,
				usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
			});
		}
		this.profileActive = !0, this.profileCapturing = !0;
	}
	disableProfile() {
		this.profileActive = !1, this.profileCapturing = !1;
	}
	resetProfileSamples() {
		this.profileSamples.clear(), this.profileForwardTotals_ns.length = 0, this.profileOverflow = !1;
	}
	enableCpuProfile() {
		this.cpuProfileActive = !0, this.cpuProfileCapturing = !0;
	}
	disableCpuProfile() {
		this.cpuProfileActive = !1, this.cpuProfileCapturing = !1;
	}
	resetCpuProfileSamples() {
		this.cpuPhaseSamples.clear(), this.cpuForwardSamples_ms.length = 0, this.cpuForwardWriteBufferCounts.length = 0, this.cpuForwardPassEncodeCounts.length = 0, this.cpuPrevForwardEnd_ms = 0;
	}
	getCpuProfileReport() {
		let e = [];
		for (let [t, n] of this.cpuPhaseSamples) {
			if (n.length === 0) continue;
			let r = n.slice().sort((e, t) => e - t), i = r[Math.floor(r.length * .5)], a = r[Math.min(r.length - 1, Math.floor(r.length * .95))], o = 0;
			for (let e of n) o += e;
			e.push({
				phase: t,
				samples_n: n.length,
				p50_ms: i,
				p95_ms: a,
				mean_ms: o / n.length,
				total_ms: o
			});
		}
		e.sort((e, t) => t.p50_ms - e.p50_ms);
		let t = (e) => {
			if (e.length === 0) return 0;
			let t = e.slice().sort((e, t) => e - t);
			return t[Math.floor(t.length * .5)];
		};
		return {
			perPhase: e,
			forwardMedian_ms: t(this.cpuForwardSamples_ms),
			forwardSamples_n: this.cpuForwardSamples_ms.length,
			writeBufferCountMedian: t(this.cpuForwardWriteBufferCounts),
			passEncodeCountMedian: t(this.cpuForwardPassEncodeCounts)
		};
	}
	getProfileReport() {
		let e = [];
		for (let [t, n] of this.profileSamples) {
			if (n.length === 0) continue;
			let r = n.slice().sort((e, t) => e - t), i = r[Math.floor(r.length * .5)], a = r[Math.min(r.length - 1, Math.floor(r.length * .95))], o = 0;
			for (let e of n) o += e;
			e.push({
				label: t,
				samples_n: n.length,
				p50_ns: i,
				p95_ns: a,
				mean_ns: o / n.length,
				total_ns: o
			});
		}
		e.sort((e, t) => t.p50_ns - e.p50_ns);
		let t = this.profileForwardTotals_ns.slice().sort((e, t) => e - t);
		return {
			perLabel: e,
			forwardMedian_ns: t.length > 0 ? t[Math.floor(t.length * .5)] : 0,
			forwardSamples_n: this.profileForwardTotals_ns.length,
			overflow: this.profileOverflow
		};
	}
	beginPass(e, t) {
		if (this.cpuProfileActive && this.cpuProfileCapturing && this.cpuPassCountThisForward++, !this.profileActive || !this.profileCapturing || !this.profileQuerySet) return e.beginComputePass();
		let n = this.profileSlotCursor;
		return n + 2 > this.profileCapability.querySlots ? (this.profileOverflow = !0, e.beginComputePass()) : (this.profileSlotCursor = n + 2, this.profilePassLabels.push(t), e.beginComputePass({ timestampWrites: {
			querySet: this.profileQuerySet,
			beginningOfPassWriteIndex: n,
			endOfPassWriteIndex: n + 1
		} }));
	}
	appendProfileResolve(e) {
		if (!this.profileActive || !this.profileCapturing || !this.profileQuerySet || !this.profileResolveBuf || !this.profileStagingBuf) return !1;
		let t = this.profileSlotCursor;
		return t === 0 ? !1 : (e.resolveQuerySet(this.profileQuerySet, 0, t, this.profileResolveBuf, 0), e.copyBufferToBuffer(this.profileResolveBuf, 0, this.profileStagingBuf, 0, t * 8), !0);
	}
	async collectProfileSamples() {
		let e = this.profileStagingBuf, t = this.profilePassLabels, n = this.profileSlotCursor;
		if (this.profileSlotCursor = 0, this.profilePassLabels = [], n === 0) return;
		try {
			await e.mapAsync(GPUMapMode.READ, 0, n * 8);
		} catch (e) {
			console.warn(`profile staging map failed: ${e}`);
			return;
		}
		let r = new BigInt64Array(e.getMappedRange(0, n * 8).slice(0));
		e.unmap();
		let i = 0;
		for (let e = 0; e < t.length; e++) {
			let n = r[e * 2], a = r[e * 2 + 1];
			if (a <= n) continue;
			let o = Number(a - n);
			i += o;
			let s = t[e], c = this.profileSamples.get(s);
			c || (c = [], this.profileSamples.set(s, c)), c.push(o);
		}
		i > 0 && this.profileForwardTotals_ns.push(i);
	}
	dispose() {
		let e = (e) => {
			e && e.destroy();
		};
		if (this.profileResolveBuf && e(this.profileResolveBuf), this.profileStagingBuf && e(this.profileStagingBuf), this.profileQuerySet && this.profileQuerySet.destroy(), e(this.modelBuffers?.tokenEmbed), e(this.modelBuffers?.finalNorm), this.modelBuffers?.layers) for (let t of this.modelBuffers.layers) for (let n of Object.values(t)) e(n);
		if (this.workBuffers) for (let t of Object.values(this.workBuffers)) if (Array.isArray(t)) for (let n of t) e(n);
		else e(t);
		if (this.kvCaches) for (let t of this.kvCaches) e(t.k), e(t.v);
		if (this.uniformBuffers) for (let t of Object.values(this.uniformBuffers)) if (Array.isArray(t)) for (let n of t) e(n);
		else e(t);
		this.device?.destroy();
	}
};
//#endregion
//#region src/index.ts
async function U(e = {}) {
	let t = new H(e);
	return await t.init(e), t;
}
//#endregion
export { E as PROFILES, w as appleMSeries, U as createGemmaEngine, T as generic, C as nvidiaBlackwell, S as overrideProfile, x as rowsPerWorkgroupFor, O as selectDeviceProfile };
