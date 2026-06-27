// SwiGLU activation: out = SiLU(gate) * up, where SiLU(x) = x * sigmoid(x).
// Mirrors gelu_mul.wgsl (same binding layout / dispatch) so the FFN path can
// switch GELU↔SiLU on a config flag with no other plumbing change. Qwen3 (and
// the llama/Qwen MLP family) use SiLU here where Gemma 4 uses tanh-GELU.
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= size) { return; }
  let x = gate[i];
  // SiLU(x) = x * sigmoid(x) = x / (1 + e^-x). Clamp the exponent argument to
  // keep parity with gelu_mul.wgsl's overflow-guard discipline on f32.
  let silu = x / (1.0 + exp(-clamp(x, -88.0, 88.0)));
  output[i] = silu * up[i];
}
