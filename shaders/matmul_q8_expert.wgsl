// Expert-indexed variant of matmul_q8_mr for MoE routed experts. Same
// dispatch/selection contract as matmul_q4k_expert (z = top-k slot, expert id
// from `sel`, per-slot output regions, optional per-slot input regions) with
// the in-shader Q8_0 weight format (int8 packed 4-per-u32 + f16 scale per
// 32-block). Used for `ffn_down_exps`, whose N (= moe_intermediate 896) is not
// a multiple of 256 — the q4k super-block requirement — but is a multiple of
// 32, the q8 block. (Bonus: the source tensor is Q5_0, so q8 storage is the
// higher-fidelity round-trip anyway. The shexp/dense downs share the q8 path
// for uniformity even though shexp's N=1792 would satisfy q4k.)
enable f16;
struct Params { M: u32, N: u32, rows_per_expert: u32, input_per_slot: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> qscale: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> sel: array<u32>;

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
  let slot = wg.z;
  let m0 = (wg.y * ng.x + wg.x) * R;
  let tid = lid.x;
  let N = params.N;
  let M = params.M;
  let nblocks = N / 32u;
  let rowBase = sel[slot] * params.rows_per_expert;
  let inBase = select(0u, slot * N, params.input_per_slot == 1u);

  let mc0 = rowBase + min(m0, M - 1u);
  let mc1 = rowBase + min(m0 + 1u, M - 1u);
  let mc2 = rowBase + min(m0 + 2u, M - 1u);
  let mc3 = rowBase + min(m0 + 3u, M - 1u);
  // N % 32 == 0 ⇒ every row base m·N is a multiple of 4, so the packed int8
  // word index decomposes as (m·N)>>2 + (k>>2) with lane k&3.
  let q0 = (mc0 * N) >> 2u;   let s0 = mc0 * nblocks;
  let q1 = (mc1 * N) >> 2u;   let s1 = mc1 * nblocks;
  let q2 = (mc2 * N) >> 2u;   let s2 = mc2 * nblocks;
  let q3 = (mc3 * N) >> 2u;   let s3 = mc3 * nblocks;

  var acc0: f32 = 0.0;
  var acc1: f32 = 0.0;
  var acc2: f32 = 0.0;
  var acc3: f32 = 0.0;

  var k: u32 = tid;
  loop {
    if (k >= N) { break; }
    let inp = input[inBase + k];
    let blk = k >> 5u;
    let widx = k >> 2u;
    let lane = k & 3u;

    acc0 = acc0 + f32(unpack_i8(qweight[q0 + widx], lane)) * f32(qscale[s0 + blk]) * inp;
    acc1 = acc1 + f32(unpack_i8(qweight[q1 + widx], lane)) * f32(qscale[s1 + blk]) * inp;
    acc2 = acc2 + f32(unpack_i8(qweight[q2 + widx], lane)) * f32(qscale[s2 + blk]) * inp;
    acc3 = acc3 + f32(unpack_i8(qweight[q3 + widx], lane)) * f32(qscale[s3 + blk]) * inp;
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
      output[slot * M + m] = partials[tid * WG];
    }
  }
}
