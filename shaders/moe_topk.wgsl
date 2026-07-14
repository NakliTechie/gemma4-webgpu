// DeepSeek-V2 MoE router: softmax over `n` router logits, greedy top-k, gate
// weights = raw softmax probabilities (norm==0, the Unlimited-OCR case) or
// renormalized to sum 1 (norm==1), then × scale (routed_scaling_factor).
//
// Output `sel` layout: k expert ids (u32) followed by k gate weights
// (f32 bitcast to u32). Consumed by matmul_*_expert (ids) and moe_accum
// (weights) — keeping selection on-GPU avoids a per-layer readback stall.
//
// Single-workgroup dispatch. The max/sum reductions are parallel over 256
// lanes; the greedy k-pass selection runs on thread 0 (n ≤ 256 and k ≤ 8 —
// a serial scan over ≤ 2048 elements is noise next to the expert GEMVs).
// Ties select the lowest index (strict `>`), matching torch.topk's stable
// ordering.
struct Params { n: u32, k: u32, norm: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> sel: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> scratch: array<f32, 256>;
var<workgroup> wgMax: f32;
var<workgroup> wgSum: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let n = params.n;

  // Parallel max. (Index clamped so lanes ≥ n never read OOB; their value
  // is discarded by the select either way.)
  scratch[tid] = select(-3.402823e38, logits[min(tid, n - 1u)], tid < n);
  workgroupBarrier();
  var stride: u32 = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { scratch[tid] = max(scratch[tid], scratch[tid + stride]); }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) { wgMax = scratch[0]; }
  workgroupBarrier();

  // Parallel sum of exp(logit - max).
  scratch[tid] = select(0.0, exp(logits[min(tid, n - 1u)] - wgMax), tid < n);
  workgroupBarrier();
  stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (tid < stride) { scratch[tid] = scratch[tid] + scratch[tid + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (tid == 0u) { wgSum = scratch[0]; }
  workgroupBarrier();

  if (tid == 0u) {
    let k = params.k;
    var taken: array<bool, 256>;
    for (var i: u32 = 0u; i < n; i = i + 1u) { taken[i] = false; }
    var wsum: f32 = 0.0;
    for (var j: u32 = 0u; j < k; j = j + 1u) {
      var best: f32 = -1.0;
      var bi: u32 = 0u;
      for (var i: u32 = 0u; i < n; i = i + 1u) {
        if (!taken[i]) {
          let p = exp(logits[i] - wgMax) / wgSum;
          if (p > best) { best = p; bi = i; }
        }
      }
      taken[bi] = true;
      sel[j] = bi;
      sel[k + j] = bitcast<u32>(best);
      wsum = wsum + best;
    }
    if (params.norm == 1u) {
      for (var j: u32 = 0u; j < k; j = j + 1u) {
        let w = bitcast<f32>(sel[k + j]);
        sel[k + j] = bitcast<u32>(w / wsum * params.scale);
      }
    } else if (params.scale != 1.0) {
      for (var j: u32 = 0u; j < k; j = j + 1u) {
        let w = bitcast<f32>(sel[k + j]);
        sel[k + j] = bitcast<u32>(w * params.scale);
      }
    }
  }
}
