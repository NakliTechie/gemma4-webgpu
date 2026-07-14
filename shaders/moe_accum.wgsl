// MoE combine: output[h] += Σ_{j<k} w_j · slots[j·H + h]
//
// `slots` holds each top-k expert's down-projection output in its own H-sized
// region (written by matmul_*_expert with per-slot output offsets); the gate
// weights live in the second half of the selection buffer (f32 bitcast in
// sel[k..2k], written by moe_topk). `output` already contains the
// shared-expert FFN result (computed through the standard dense-FFN path via
// tensor aliasing), so the += here lands the full DeepSeek-V2 MoE sum
// y = shared(x) + Σ w_j·E_j(x) in place, ready for the residual add.
struct Params { h: u32, k: u32 }
@group(0) @binding(0) var<storage, read> slots: array<f32>;
@group(0) @binding(1) var<storage, read> sel: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  if (h >= params.h) { return; }
  var acc: f32 = 0.0;
  for (var j: u32 = 0u; j < params.k; j = j + 1u) {
    let w = bitcast<f32>(sel[params.k + j]);
    acc = acc + w * slots[j * params.h + h];
  }
  output[h] = output[h] + acc;
}
