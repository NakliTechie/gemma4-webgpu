// In-shader Q8_0 matmul (GEMV): output[m] = Σ_k dequant(qweight[m,k]) · input[k].
//
// Weights are kept QUANTIZED in GPU memory (1 byte/weight vs F16's 2) and
// dequantized inside the dot-product loop — the memory lever that lets larger
// models fit. Symmetric int8 with one f16 scale per 32-element block (GGUF
// Q8_0 layout, re-derived from the source weights at load):
//   qweight : int8, row-major [M, N], packed 4-per-u32 over the flattened array
//   qscale  : f16,  [M, N/32]  (one scale per 32-wide block per row)
// dequant(w[m,k]) = f32(int8) · scale[m, k/32].
//
// Bandwidth-bound GEMV: halving weight bytes read should also help tps, the
// same reason ORT's MatMulNBits / the LFM2 engine keep weights quantized.
enable f16;
struct Params { M: u32, N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> qscale: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> partials: array<f32, 256>;

// Extract the `lane`-th signed int8 (little-endian) from a packed u32.
fn unpack_i8(packed: u32, lane: u32) -> i32 {
  let b = (packed >> (lane * 8u)) & 0xffu;
  return select(i32(b), i32(b) - 256, b >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wg.y * ng.x + wg.x;
  if (m >= params.M) { return; }
  let tid = lid.x;
  let N = params.N;
  let nblocks = N / 32u;
  let row_q_base = m * N;        // flattened int8 element index for row m
  let row_s_base = m * nblocks;  // scale index for row m

  var acc: f32 = 0.0;
  var k: u32 = tid;
  loop {
    if (k >= N) { break; }
    let gi = row_q_base + k;
    let qv = unpack_i8(qweight[gi >> 2u], gi & 3u);
    let scale = f32(qscale[row_s_base + (k >> 5u)]);
    acc = acc + f32(qv) * scale * input[k];
    k = k + WG;
  }
  partials[tid] = acc;
  workgroupBarrier();

  var stride: u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partials[tid] = partials[tid] + partials[tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid == 0u) {
    output[m] = partials[0];
  }
}
