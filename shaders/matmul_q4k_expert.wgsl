// Expert-indexed variant of matmul_q4k_mr for MoE routed experts. The weight
// buffer packs ALL experts contiguously ([expert][rows_per_expert][N] in the
// engine's in-shader q4k layout); the expert id for this dispatch's z-slice is
// read from the on-GPU selection buffer (`sel`, written by moe_topk) — no CPU
// readback between router and expert GEMVs.
//
// Dispatch: (ceil(M/R), 1, k) — z = slot index into the top-k selection.
//   row base   = sel[slot] * rows_per_expert
//   output     = output[slot * M + m]           (per-slot output regions)
//   input      = input[k]                        (input_per_slot == 0: shared
//                input, e.g. the normed hidden for gate/up), or
//                input[slot * N + k]             (input_per_slot == 1: per-slot
//                input regions, e.g. the activated mul for down-proj).
// Same MR4 body as matmul_q4k_mr otherwise (R=4 rows/wg, shared tree-reduce).
enable f16;
struct Params { M: u32, N: u32, rows_per_expert: u32, input_per_slot: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> qmeta: array<f16>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> sel: array<u32>;

const WG: u32 = 256u;
const R: u32 = 4u;
var<workgroup> partials: array<f32, 1024>; // 256 * 4

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(num_workgroups) ng: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let slot = wg.z;
  let m0 = (wg.y * ng.x + wg.x) * R;
  let tid = lid.x;
  let N = params.N;
  let M = params.M;
  let nsblk = N / 256u;
  let rowBase = sel[slot] * params.rows_per_expert;
  let inBase = select(0u, slot * N, params.input_per_slot == 1u);

  // Clamp rows to a valid index so out-of-range rows (M % 4 != 0) read
  // in-bounds; their output is never written below.
  let mc0 = rowBase + min(m0, M - 1u);
  let mc1 = rowBase + min(m0 + 1u, M - 1u);
  let mc2 = rowBase + min(m0 + 2u, M - 1u);
  let mc3 = rowBase + min(m0 + 3u, M - 1u);
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
    let inp = input[inBase + k];
    let sb = k >> 8u;
    let sub = (k & 255u) >> 5u;
    let moff = sb * 16u + sub;
    let shift = (k & 7u) * 4u;
    let widx = k >> 3u;

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
      output[slot * M + m] = partials[tid * WG];
    }
  }
}
