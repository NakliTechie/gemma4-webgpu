// In-shader 4-bit matmul (GEMV): output[m] = Σ_k dequant(qweight[m,k]) · input[k].
//
// The 4-bit memory lever — the largest one. Weights are kept at 4 bits/value in
// GPU memory (vs F16's 16, Q8's 8) and dequantized inside the dot-product loop,
// the same idea as ORT's MatMulNBits. This is what lets the 4–8B class fit.
//
// Block-affine 4-bit, mirroring GGUF Q4_K's *structure* (256-element super-block
// split into 8 sub-blocks of 32, each sub-block an independent 4-bit affine grid)
// but keeping each sub-block's scale+min as f16 rather than packing them to 6 bits
// via a super-scale. Trading ~0.5 bit/weight of packing for a cleaner, strictly
// higher-precision representation than GGUF Q4_K — so a Q4_K *source* tensor round-
// trips near-losslessly, and only genuinely-higher-precision source tensors (Q6_K)
// take a real down-quant:
//   qweight : 4-bit unsigned, row-major [M, N], packed 8-per-u32 over the flat array
//   qmeta    : f16, [M, N/256, 16] — per super-block, 8 scales then 8 mins
// dequant(w[m,k]) = scale[m, sb, sub] · f32(nibble) + min[m, sb, sub],
//   where sb = (k mod N)/256, sub = (k mod 256)/32, nibble ∈ [0,15].
enable f16;
struct Params { M: u32, N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> qmeta: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = wg.y * ng.x + wg.x;
  if (m >= params.M) { return; }
  let tid = lid.x;
  let N = params.N;
  let nsblk = N / 256u;             // super-blocks per row
  let row_q_base = m * N;           // flat 4-bit element index for row m
  let row_qmeta_base = m * nsblk * 16u;  // f16 qmeta index for row m

  var acc: f32 = 0.0;
  var k: u32 = tid;
  loop {
    if (k >= N) { break; }
    // Unpack the 4-bit weight: 8 nibbles per u32, little-endian nibble order.
    let fi = row_q_base + k;
    let nib = (qweight[fi >> 3u] >> ((fi & 7u) * 4u)) & 0xfu;
    // Locate the sub-block this element belongs to and read its affine pair.
    let sb = k >> 8u;              // k / 256
    let sub = (k & 255u) >> 5u;    // (k % 256) / 32  ∈ [0,8)
    let mbase = row_qmeta_base + sb * 16u;
    let scale = f32(qmeta[mbase + sub]);
    let mn = f32(qmeta[mbase + 8u + sub]);
    acc = acc + (scale * f32(nib) + mn) * input[k];
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
