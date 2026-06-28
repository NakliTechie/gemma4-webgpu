// Multi-row variant of matmul_q4k (R=4 output rows per workgroup), the speed
// half of the 4-bit lever. Mirrors matmul_quant_mr4's structure — read input[k]
// once, multiply into R per-row accumulators, share the 8-level tree-reduce —
// but each row dequantizes its own 4-bit block-affine weight in the loop.
// Caller dispatches ceil(M/R) workgroups. Bindings/Params match matmul_q4k
// exactly, so the bind-group layout is structurally identical (only the pipeline
// object differs). The block-affine sub-position (sb, sub) is the same for all R
// rows at a given k, so it's computed once per iteration.
//
// Why this is the win at batch-1 decode: the GEMV is launch-overhead/occupancy-
// bound, not bandwidth-bound (~9 GB/s effective vs ~200+ GB/s peak), so 4× fewer
// workgroups + amortized input/reduce buys more than any memory-traffic tweak.
enable f16;
struct Params { M: u32, N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> qmeta: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;
const R: u32 = 4u;
var<workgroup> partials: array<f32, 1024>; // 256 * 4

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let m0 = (wg.y * ng.x + wg.x) * R;
  let tid = lid.x;
  let N = params.N;
  let M = params.M;
  let nsblk = N / 256u;

  // Clamp each row to a valid index so out-of-range rows (when M % 4 != 0) read
  // in-bounds data; their output is simply never written below.
  let mc0 = min(m0, M - 1u);
  let mc1 = min(m0 + 1u, M - 1u);
  let mc2 = min(m0 + 2u, M - 1u);
  let mc3 = min(m0 + 3u, M - 1u);
  let q0 = mc0 * N;       let s0 = mc0 * nsblk * 16u;
  let q1 = mc1 * N;       let s1 = mc1 * nsblk * 16u;
  let q2 = mc2 * N;       let s2 = mc2 * nsblk * 16u;
  let q3 = mc3 * N;       let s3 = mc3 * nsblk * 16u;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;

  var k: u32 = tid;
  loop {
    if (k >= N) { break; }
    let inp = input[k];
    let sb = k >> 8u;
    let sub = (k & 255u) >> 5u;
    let moff = sb * 16u + sub;       // scale offset within a row's meta
    let shift = (k & 7u) * 4u;
    let widx = k >> 3u;              // u32-word offset within a row's quants

    let nib0 = (qweight[(q0 >> 3u) + widx] >> shift) & 0xfu;
    acc0 = acc0 + (f32(qmeta[s0 + moff]) * f32(nib0) + f32(qmeta[s0 + moff + 8u])) * inp;
    let nib1 = (qweight[(q1 >> 3u) + widx] >> shift) & 0xfu;
    acc1 = acc1 + (f32(qmeta[s1 + moff]) * f32(nib1) + f32(qmeta[s1 + moff + 8u])) * inp;
    let nib2 = (qweight[(q2 >> 3u) + widx] >> shift) & 0xfu;
    acc2 = acc2 + (f32(qmeta[s2 + moff]) * f32(nib2) + f32(qmeta[s2 + moff + 8u])) * inp;
    let nib3 = (qweight[(q3 >> 3u) + widx] >> shift) & 0xfu;
    acc3 = acc3 + (f32(qmeta[s3 + moff]) * f32(nib3) + f32(qmeta[s3 + moff + 8u])) * inp;
    k = k + WG;
  }

  partials[0u * WG + tid] = acc0;
  partials[1u * WG + tid] = acc1;
  partials[2u * WG + tid] = acc2;
  partials[3u * WG + tid] = acc3;
  workgroupBarrier();

  var stride: u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) {
      partials[0u * WG + tid] = partials[0u * WG + tid] + partials[0u * WG + tid + stride];
      partials[1u * WG + tid] = partials[1u * WG + tid] + partials[1u * WG + tid + stride];
      partials[2u * WG + tid] = partials[2u * WG + tid] + partials[2u * WG + tid + stride];
      partials[3u * WG + tid] = partials[3u * WG + tid] + partials[3u * WG + tid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (tid < R) {
    let m = m0 + tid;
    if (m < M) {
      output[m] = partials[tid * WG];
    }
  }
}
