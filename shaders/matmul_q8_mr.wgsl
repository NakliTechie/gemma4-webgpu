// Multi-row variant of matmul_q8 (R=4 output rows per workgroup) — the Q8 speed
// path, mirroring matmul_q4k_mr. Read input[k] once, dequantize 4 rows' int8
// weights in the loop, share the 8-level tree-reduce. Caller dispatches
// ceil(M/R) workgroups; bindings/Params match matmul_q8 exactly so the
// bind-group layout is structurally identical (only the pipeline differs).
// N % 32 == 0 (Q8 block) ⇒ each row base m·N is a multiple of 4, so the packed
// int8 word index decomposes as (m·N>>2)+(k>>2) with lane k&3.
enable f16;
struct Params { M: u32, N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> qscale: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 256u;
const R: u32 = 4u;
var<workgroup> partials: array<f32, 1024>; // 256 * 4

fn unpack_i8(packed: u32, lane: u32) -> i32 {
  let b = (packed >> (lane * 8u)) & 0xffu;
  return select(i32(b), i32(b) - 256, b >= 128u);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let m0 = (wg.y * ng.x + wg.x) * R;
  let tid = lid.x;
  let N = params.N;
  let M = params.M;
  let nblocks = N / 32u;

  // Clamp each row to a valid index so out-of-range rows (M % 4 != 0) read
  // in-bounds; their output is never written below.
  let mc0 = min(m0, M - 1u);
  let mc1 = min(m0 + 1u, M - 1u);
  let mc2 = min(m0 + 2u, M - 1u);
  let mc3 = min(m0 + 3u, M - 1u);
  let q0 = mc0 * N;  let s0 = mc0 * nblocks;
  let q1 = mc1 * N;  let s1 = mc1 * nblocks;
  let q2 = mc2 * N;  let s2 = mc2 * nblocks;
  let q3 = mc3 * N;  let s3 = mc3 * nblocks;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;

  var k: u32 = tid;
  loop {
    if (k >= N) { break; }
    let inp = input[k];
    let widx = k >> 2u;
    let lane = k & 3u;
    let sblk = k >> 5u;
    acc0 = acc0 + f32(unpack_i8(qweight[(q0 >> 2u) + widx], lane)) * f32(qscale[s0 + sblk]) * inp;
    acc1 = acc1 + f32(unpack_i8(qweight[(q1 >> 2u) + widx], lane)) * f32(qscale[s1 + sblk]) * inp;
    acc2 = acc2 + f32(unpack_i8(qweight[(q2 >> 2u) + widx], lane)) * f32(qscale[s2 + sblk]) * inp;
    acc3 = acc3 + f32(unpack_i8(qweight[(q3 >> 2u) + widx], lane)) * f32(qscale[s3 + sblk]) * inp;
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
    if (m < M) { output[m] = partials[tid * WG]; }
  }
}
