import { SHADERS } from './shaders.js';
import { GGUFParser, tensorByteSize, f32ToF16Array, f32ToF16 } from './gguf.js';
import { Tokenizer } from './tokenizer.js';
import { buildChatPrompt } from './conversation.js';
import { defaultConfig, configFromGGUF } from './config/gemma4-e2b.js';
import { configFromGGUFQwen3 } from './config/qwen3-4b.js';
import { configFromGGUFDeepseekOcr } from './config/deepseek-ocr.js';
import { selectDeviceProfile, rowsPerWorkgroupFor, type TuningProfile } from './tuning/index.js';
import { RangedReader } from './ranged-reader.js';
import {
  getRopeTheta,
  getHeadDim,
  getIntermediateSize,
  isSwaLayer,
  isKvProducerLayer,
  isMoELayer,
  type GemmaConfig,
  type GemmaEngine,
  type GemmaEngineOptions,
  type GenerateOptions,
  type ProgressInfo,
  type ConversationTurn,
  type GGUFTensor,
  type ModelBuffers,
  type WorkBuffers,
  type KVCache,
  type UniformBuffers,
  type BindGroupCache,
  type LayerBindGroups,
  type CapturePoint,
  type ProfileCapability,
  type ProfileReport,
  type CpuProfileReport,
  type LoadTimings,
} from './types.js';


// Model catalog. Gemma 4 E2B is the target. Q4_K_M (llama.cpp's mixed-quant
// strategy) is CPU-dequantized to F16 at load, so the matmul shader handles
// one storage layout regardless of source quant. Re-verify the URL via
// `scripts/audit_gguf.mjs` if it ever changes.
const MODELS: Record<string, string> = {
  'e2b': 'https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf',
};

// Above this element count, dequant a tensor to F16 in chunks rather than via one
// giant F32 intermediate. ~256M elems = a 1 GB F32 buffer; the 8B token_embd
// (622M) would otherwise need 2.49 GB in one shot and blow V8's alloc ceiling.
const F16_CHUNK_THRESHOLD = 256 * 1024 * 1024;

// Float16Array is available in Chrome 135+ but isn't yet surfaced by the
// TS lib. The diagnostic readback helpers use it to view an ArrayBuffer as
// half-precision without a manual bit-pattern loop. Only the diagnostic
// paths reach this constructor; they run on the same platform that the
// engine already requires `shader-f16` for, so a runtime check is
// unnecessary.
type Float16ArrayCtor = new (
  source: ArrayBufferLike | number | ArrayLike<number>,
  byteOffset?: number,
  length?: number,
) => ArrayBufferView & { readonly length: number; [index: number]: number };
const F16A = (globalThis as unknown as { Float16Array: Float16ArrayCtor }).Float16Array;

// Per-layer tensor families in the GGUF.
//  - Matmul weights (attn_q/k/v/output, ffn_gate/up/down): big, quantized,
//    CPU-dequantized to F16 at upload via `tensorToF16`.
//  - F32 weights: small norms, per-layer PLE transforms, residual-gain scales.
const LAYER_MATMUL_WEIGHT_NAMES = [
  'attn_q', 'attn_k', 'attn_v', 'attn_output',
  'ffn_gate', 'ffn_up', 'ffn_down',
];
const LAYER_F32_WEIGHT_NAMES = [
  'attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm',
  'post_attention_norm', 'post_ffw_norm',
  // Per-layer PLE + residual-gain weights:
  'inp_gate', 'proj', 'post_norm', 'layer_output_scale',
];

// Gemma 4 global tensors. All uploaded as F16 on GPU regardless of source quant.
// Shapes confirmed via scripts/audit_gguf.mjs against the live Q4_K_M file:
//   per_layer_model_proj  [1536, 8960]    source BF16     — ~26 MB as F16 on GPU
//   per_layer_proj_norm   [256]           source F32      — ~512 B
//   rope_freqs            [256]           source F32      — ~512 B
//   per_layer_token_embd  [8960, 262144]  source Q5_K     — uploaded separately, split 35 ways
const GLOBAL_F32_TENSORS = ['per_layer_proj_norm', 'rope_freqs', 'per_layer_model_proj'];

export class GemmaEngineImpl implements GemmaEngine {
  config: GemmaConfig;
  tuning!: TuningProfile;
  tuningReason!: string;
  private adapter!: GPUAdapter;
  private device!: GPUDevice;
  private pipelines!: Record<string, GPUComputePipeline>;
  /** Cached from `tuning.matmul.rowsPerWorkgroupByKernel['ffn.linearGateUp']`.
   * `true` selects the MR4 pipeline (4-row unrolled); `false` uses the
   * scalar matmulQuant path with `dispatchMatmul(I)`. */
  private mr4ForFfn!: boolean;
  /** Output rows produced per workgroup by the active layer-matmul kernel: 4 for
   *  the multi-row kernels (q4k/q8 MR always; f16 when mr4 is enabled), else 1
   *  for the scalar path. Drives `dispatchMatmulRows` (ceil(M/rows) workgroups). */
  private matmulRowsPerWg = 1;
  /** Weight storage for the layer matmul weights. 'f16' (default) dequantizes
   *  to F16 at load; 'q8' keeps them as in-shader Q8_0 (int8 + per-32-block f16
   *  scale) for ~2× less GPU memory; 'q4k' keeps them as in-shader 4-bit
   *  block-affine (per-32 4-bit + f16 scale/min) for ~4× less. lmHead/embeddings/
   *  norms stay F16 in all modes. */
  private weightQuant: 'f16' | 'q8' | 'q4k' = 'f16';
  private modelBuffers!: ModelBuffers;
  private workBuffers!: WorkBuffers;
  private uniformBuffers!: UniformBuffers;
  private kvCaches!: KVCache[];
  private bindGroupCache!: BindGroupCache;
  private tokenizer!: Tokenizer;

  private conversationHistory: ConversationTurn[] = [];
  private kvPosition: number = 0;
  /** Optional system instruction, emitted as a `developer` turn preamble. */
  private _systemPrompt: string | null = null;
  /** Timing stats from the most recent `generate()` call. See the
   *  `lastGenerateStats` getter on the `GemmaEngine` interface. */
  private _lastGenerateStats: import('./types.js').LastGenerateStats | null = null;
  private onProgress?: (progress: ProgressInfo) => void;
  private deviceLost: boolean = false;

  // ─── Profiling state ────────────────────────────────────────────
  // Single pre-allocated query set + staging buffers reused across forward
  // passes. The attach/detach cost is just mutating a descriptor field; no
  // per-pass allocation. `profileActive` is the master switch read by the
  // per-pass helper; `profileCapturing` is what actually gates timestampWrites
  // (so we can cheaply skip prefill by leaving active=true capturing=false).
  private profileCapability: ProfileCapability = {
    timestampQuerySupported: false,
    timestampQueryOnAdapterButUnrequested: false,
    querySlots: 0,
  };
  private profileActive = false;
  private profileCapturing = false;
  private profileQuerySet: GPUQuerySet | null = null;
  private profileResolveBuf: GPUBuffer | null = null;
  private profileStagingBuf: GPUBuffer | null = null;
  private profileSlotCursor = 0;
  private profilePassLabels: string[] = [];
  private profileSamples: Map<string, number[]> = new Map();
  private profileForwardTotals_ns: number[] = [];
  private profileOverflow = false;
  private static readonly PROFILE_QUERY_SLOTS = 2048;

  // ─── CPU-side phase profiling ───────────────────────────────────
  // Independent from GPU profiling. Measures CPU wall time around writeBuffer
  // block, pass encoding, encoder.finish, queue.submit, mapAsync wait, and
  // readback processing. `cpuProfileCapturing` mirrors `profileCapturing` so
  // prefill can disable capture without touching the master switch.
  private cpuProfileActive = false;
  private cpuProfileCapturing = false;
  private cpuPhaseSamples: Map<string, number[]> = new Map();
  private cpuForwardSamples_ms: number[] = [];
  private cpuForwardWriteBufferCounts: number[] = [];
  private cpuForwardPassEncodeCounts: number[] = [];
  // Per-forward scratch — written by encodeTransformerPass, read by forwardPassAndGetToken.
  private cpuWbCountThisForward = 0;
  private cpuPassCountThisForward = 0;
  // performance.now() at the end of the previous forwardPassAndGetToken, so we
  // can attribute time spent OUTSIDE the forward pass (generator loop, yield
  // resumption, tokenizer.decode, consumer-side work) as a phase.
  private cpuPrevForwardEnd_ms = 0;

  // Depth-2 pipelined decode (Option B) ping-pong index for the argmax
  // readback pool. Flipped each greedy-fast submit so two in-flight forwards
  // map distinct buffers (mapAsync forbids concurrent maps on one buffer).
  private argmaxReadbackIdx: 0 | 1 = 0;

  // Persisted GGUF metadata for post-load diagnostics (weight-layout probes).
  // modelFile is the resolved URL used for Range refetches; undefined when loaded
  // from an in-memory buffer. ggufTensors + ggufDataOffset let us look up a named
  // tensor's absolute file offset, source type, and dims after upload.
  private modelFile?: string;
  private ggufTensors?: GGUFTensor[];
  private ggufDataOffset?: number;

  // One-shot load-phase timings, populated by init() and the two upload
  // paths. Internal field is writable; exposed as a readonly getter so
  // the engine is the only place mutating it.
  private _loadTimings: LoadTimings | null = null;
  get loadTimings(): LoadTimings | null {
    return this._loadTimings;
  }

  constructor(options: GemmaEngineOptions) {
    this.config = defaultConfig();
    if (options.contextLength) {
      this.config.context_length = options.contextLength;
    }
    this.onProgress = options.onProgress;
  }

  async init(options: GemmaEngineOptions): Promise<void> {
    const modelKey = options.model || 'e2b';
    const modelFile = MODELS[modelKey] || modelKey;

    const t0 = performance.now();
    this._loadTimings = {
      supportsRange: false,
      modelFile,
      startedAt_ms: t0,
      totalMs: 0,
      phases: {},
    };
    // Mutable closure over the last marker — each `mark()` records the
    // delta since the previous mark into `phases[name]` and advances.
    let lastMark = t0;
    const mark = (name: keyof LoadTimings['phases']): void => {
      const now = performance.now();
      this._loadTimings!.phases[name] = now - lastMark;
      lastMark = now;
    };

    await this.initWebGPU();
    mark('initWebGPU_ms');

    // Select tuning profile. Pipelines and bind-group layouts downstream
    // depend on `this.mr4ForFfn`, so profile resolution must happen before
    // `createPipelines`.
    const selected = selectDeviceProfile(this.adapter, options.tuning);
    this.tuning = selected.profile;
    this.tuningReason = selected.reason;
    this.mr4ForFfn = rowsPerWorkgroupFor(this.tuning, 'ffn.linearGateUp') >= 4;
    this.weightQuant = options.weightQuant ?? 'f16';
    // Rows-per-workgroup for the active layer-matmul kernel. q4k/q8 ship their own
    // multi-row (R=4) kernels; f16 uses matmulQuantMR4 (R=4) when the profile
    // enables it, else the scalar 1-row path. mr4ForFfn stays meaningful only for
    // the f16 FFN bind-group layout choice (MR4 ≡ matmulQuant layout either way).
    this.matmulRowsPerWg =
      this.weightQuant === 'q4k' || this.weightQuant === 'q8' ? 4
        : this.mr4ForFfn ? 4 : 1;
    // Log the decision for post-hoc diagnosis of unexpectedly-low tps
    // on hardware the profile registry doesn't yet know about.
    console.log(
      `[gemma4-webgpu] tuning: ${this.tuning.id} (${selected.reason})${this.tuning.verified ? ' — verified' : ' — unverified'}`,
    );

    this.reportProgress(0, 1, 'Downloading header...', 'downloading');
    const HEADER_FETCH_SIZE = 20 * 1024 * 1024;
    const headerResp = await fetch(modelFile, {
      headers: { Range: `bytes=0-${HEADER_FETCH_SIZE - 1}` },
      signal: options.signal,
    });
    const supportsRange = headerResp.status === 206;
    this._loadTimings.supportsRange = supportsRange;

    if (supportsRange) {
      const headerBuf = new Uint8Array(await headerResp.arrayBuffer());
      mark('headerFetch_ms');

      const parser = new GGUFParser(headerBuf);
      const gguf = parser.parse();
      this.config = gguf.kv.has('deepseek2-ocr.block_count')
        ? configFromGGUFDeepseekOcr(gguf, options.contextLength)
        : gguf.kv.has('qwen3.block_count') || gguf.kv.has('qwen3.embedding_length')
          ? configFromGGUFQwen3(gguf, options.contextLength)
          : configFromGGUF(gguf, options.contextLength);

      this.tokenizer = new Tokenizer();
      this.tokenizer.extractFromGGUF(gguf);

      const tensors = gguf.tensors;
      const dataOffset = gguf.dataOffset;
      this.modelFile = modelFile;
      this.ggufTensors = tensors;
      this.ggufDataOffset = dataOffset;
      mark('ggufParse_ms');

      this.createPipelines();
      mark('createPipelines_ms');

      this.createUniformBuffers();
      mark('createUniformBuffers_ms');

      // uploadWeightsStreaming populates pleFetch/pleDecode/layers* phases
      // directly onto this._loadTimings — it owns its own sub-markers so
      // outer mark() doesn't get a lumped-in layer phase.
      await this.uploadWeightsStreaming(modelFile, tensors, dataOffset, options.signal);
      lastMark = performance.now();
    } else {
      const contentLength = headerResp.headers.get('content-length');
      const total = contentLength ? parseInt(contentLength) : 1100000000;
      const reader = headerResp.body!.getReader();
      const buffer = new Uint8Array(total);
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer.set(value, received);
        received += value.length;
        this.reportProgress(received, total, `Downloading model...`);
      }
      mark('fullBufferDownload_ms');

      const parser = new GGUFParser(buffer);
      const gguf = parser.parse();
      this.config = gguf.kv.has('deepseek2-ocr.block_count')
        ? configFromGGUFDeepseekOcr(gguf, options.contextLength)
        : gguf.kv.has('qwen3.block_count') || gguf.kv.has('qwen3.embedding_length')
          ? configFromGGUFQwen3(gguf, options.contextLength)
          : configFromGGUF(gguf, options.contextLength);

      this.tokenizer = new Tokenizer();
      this.tokenizer.extractFromGGUF(gguf);

      this.ggufTensors = gguf.tensors;
      this.ggufDataOffset = gguf.dataOffset;
      // modelFile left unset — full buffer load path, Range refetches won't work.
      mark('ggufParse_ms');

      this.createPipelines();
      mark('createPipelines_ms');

      this.createUniformBuffers();
      mark('createUniformBuffers_ms');

      await this.uploadWeightsFromBuffer(parser, gguf);
      await this.device.queue.onSubmittedWorkDone();
      mark('fullBufferUpload_ms');
    }

    // Architectural invariant: Gemma 4 E2B always has per-layer embeddings
    // (PLE) with a positive per_layer_input_dim. The forward pass and the
    // bind-group layout assume PLE is always active, so fail loud if the
    // GGUF metadata somehow yields 0 — cleaner than silent wrong output.
    const hasPle = this.config.arch === undefined || this.config.arch === 'gemma4';
    if (hasPle && this.config.per_layer_input_dim === 0) {
      throw new Error(
        'per_layer_input_dim = 0 — this engine is Gemma-4-E2B-specific and requires PLE. ' +
        'Check that the GGUF is a Gemma 4 variant.',
      );
    }

    this.createWorkBuffers();
    mark('createWorkBuffers_ms');

    this.createBindGroups();
    mark('createBindGroups_ms');

    this._loadTimings.totalMs = performance.now() - t0;
  }

  /** writeBuffer wrapper to handle @webgpu/types ArrayBuffer vs ArrayBufferLike strictness */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private wb(buf: GPUBuffer, offset: number, data: any): void {
    if (this.cpuProfileActive && this.cpuProfileCapturing) this.cpuWbCountThisForward++;
    this.device.queue.writeBuffer(buf, offset, data);
  }

  /** Append one sample to a CPU phase bucket. Called only when cpu profile is active. */
  private recordCpuPhase(phase: string, ms: number): void {
    let arr = this.cpuPhaseSamples.get(phase);
    if (!arr) { arr = []; this.cpuPhaseSamples.set(phase, arr); }
    arr.push(ms);
  }

  /**
   * matmulQuant dispatch helper. The rewritten shader uses one workgroup per
   * output row (M workgroups total). For M > WebGPU's default
   * maxComputeWorkgroupsPerDimension=65535 (the lmHead case where M=V=262144),
   * fall back to a sqrt-shaped 2D grid; the shader recovers the row index as
   * m = wg.y * ng.x + wg.x.
   */
  private dispatchMatmul(pass: GPUComputePassEncoder, M: number): void {
    const LIMIT = 65535;
    if (M <= LIMIT) {
      pass.dispatchWorkgroups(M, 1, 1);
      return;
    }
    const x = Math.min(LIMIT, Math.ceil(Math.sqrt(M)));
    const y = Math.ceil(M / x);
    pass.dispatchWorkgroups(x, y, 1);
  }

  /**
   * Matmul dispatch that accounts for the active matmul pipeline's rows-per-
   * workgroup: the q4k MR kernel produces 4 rows/workgroup → ceil(M/4) groups;
   * f16/q8 scalar produce 1 row → M groups (identical to dispatchMatmul). The
   * shader recovers the base row as (wg.y*ng.x + wg.x)*R, so the same 2D
   * overflow scheme applies to the workgroup count.
   */
  private dispatchMatmulRows(pass: GPUComputePassEncoder, M: number): void {
    this.dispatchMatmul(pass, Math.ceil(M / this.matmulRowsPerWg));
  }

  private reportProgress(
    loaded: number,
    total: number,
    status: string,
    kind?: ProgressInfo['kind'],
  ): void {
    if (this.onProgress) {
      this.onProgress({ loaded, total, status, kind });
    }
  }

  private async initWebGPU(): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    this.adapter = adapter;
    // shader-f16 lets us store weights as array<f16> and cuts weight VRAM in half.
    // Available on ~all WebGPU-capable GPUs today; if absent, the shaders still work
    // by reading u16-packed pairs via unpack2x16float — add that fallback when needed.
    const wantF16 = adapter.features.has('shader-f16');
    const hasTimestamp = adapter.features.has('timestamp-query');
    const requiredFeatures: GPUFeatureName[] = [];
    if (wantF16) requiredFeatures.push('shader-f16');
    if (hasTimestamp) requiredFeatures.push('timestamp-query');
    this.device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    // Confirm the feature actually landed on the device (Chrome may silently
    // drop it behind a flag; `device.features.has` is the authoritative check).
    const gotTimestamp = this.device.features.has('timestamp-query');
    this.profileCapability = {
      timestampQuerySupported: gotTimestamp,
      timestampQueryOnAdapterButUnrequested: hasTimestamp && !gotTimestamp,
      querySlots: gotTimestamp ? GemmaEngineImpl.PROFILE_QUERY_SLOTS : 0,
    };
    if (!wantF16) {
      throw new Error('shader-f16 WebGPU feature is required (your adapter reports it unavailable)');
    }
    this.device.lost.then((info) => {
      this.deviceLost = true;
      console.error(`WebGPU device lost: ${info.message} (reason: ${info.reason})`);
    });
  }

  private createPipelines(): void {
    this.pipelines = {};
    for (const [name, code] of Object.entries(SHADERS)) {
      const module = this.device.createShaderModule({ code });
      const compute: GPUProgrammableStage = { module, entryPoint: 'main' };
      // Inject pipeline-constant overrides from the active tuning profile.
      // matmulQuantMR4 declares `override R: u32 = 4;` — we pass the
      // profile's rows-per-workgroup for ffn.linearGateUp explicitly (in v1
      // only R=4 is supported by the unrolled shader body; other values
      // would require a variant shader).
      if (name === 'matmulQuantMR4') {
        const r = rowsPerWorkgroupFor(this.tuning, 'ffn.linearGateUp');
        compute.constants = { R: r >= 4 ? r : 4 };
      }
      this.pipelines[name] = this.device.createComputePipeline({
        layout: 'auto',
        compute,
      });
    }
  }

  private makeUniformMixed(values: ({ u: number } | { f: number })[]): GPUBuffer {
    const size = Math.max(values.length * 4, 16);
    const ab = new ArrayBuffer(size);
    const u32 = new Uint32Array(ab);
    const f32 = new Float32Array(ab);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if ('u' in v) u32[i] = v.u;
      else f32[i] = v.f;
    }
    const buf = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(ab));
    buf.unmap();
    return buf;
  }

  private createUniformBuffers(): void {
    const C = this.config;
    const H = C.hidden_size;
    const MAX_SEQ = C.context_length;
    const V = C.vocab_size;
    const NQH = C.num_q_heads;
    const NKH = C.num_kv_heads;

    this.uniformBuffers = {
      rmsNorm: this.makeUniformMixed([{ u: H }, { f: C.rms_norm_eps }]),
      sizeH: this.makeUniformMixed([{ u: H }]),
      // embed_scale: Gemma 4 = sqrt(hidden); Qwen3 = 1.0 (config.embedding_scale).
      embeddingLookup: this.makeUniformMixed([{ u: H }, { u: 0 }, { f: C.embedding_scale ?? Math.sqrt(H) }]),
      softmax: this.makeUniformMixed([{ u: NQH }, { u: 0 }]),
      linearQ8_V_H: this.makeUniformMixed([{ u: V }, { u: H }]),
      argmaxSize: this.makeUniformMixed([{ u: V }]),
      logitSoftcap: this.makeUniformMixed([{ u: V }, { f: C.final_logit_softcapping }]),
      // Per-layer embedding (PLE) uniforms.
      plePmProjMM: this.makeUniformMixed([{ u: C.num_layers * C.per_layer_input_dim }, { u: H }]),
      pleInpGateMM: this.makeUniformMixed([{ u: C.per_layer_input_dim }, { u: H }]),
      plePostProjMM: this.makeUniformMixed([{ u: H }, { u: C.per_layer_input_dim }]),
      pleStage1: [],
      pleGeluMulParams: [],
      pleSkipScaleAdd: this.makeUniformMixed([{ u: H }]),
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
      fusedNormRopeK: [],
    };

    // Per-layer uniforms. LOCAL (sliding) vs GLOBAL (full-attention) layers
    // differ in head_dim; producer vs consumer layers differ in
    // intermediate_size. `getHeadDim(il)` and `getIntermediateSize(il)`
    // resolve the per-layer value.
    for (let il = 0; il < C.num_layers; il++) {
      const HD = getHeadDim(il, C);
      const I = getIntermediateSize(il, C);
      const Q = NQH * HD;
      const KV = NKH * HD;
      const theta = getRopeTheta(il, C);

      this.uniformBuffers.perHeadRmsNormQ.push(
        this.makeUniformMixed([{ u: NQH }, { u: HD }, { f: C.rms_norm_eps }, { u: 0 }])
      );
      this.uniformBuffers.perHeadRmsNormK.push(
        this.makeUniformMixed([{ u: NKH }, { u: HD }, { f: C.rms_norm_eps }, { u: 0 }])
      );
      this.uniformBuffers.perHeadRmsNormV.push(
        this.makeUniformMixed([{ u: NKH }, { u: HD }, { f: C.rms_norm_eps }, { u: 0 }])
      );
      this.uniformBuffers.linearQ8_Q_H.push(this.makeUniformMixed([{ u: Q }, { u: H }]));
      this.uniformBuffers.linearQ8_KV_H.push(this.makeUniformMixed([{ u: KV }, { u: H }]));
      this.uniformBuffers.linearQ8_H_Q.push(this.makeUniformMixed([{ u: H }, { u: Q }]));
      this.uniformBuffers.linearQ8_I_H.push(this.makeUniformMixed([{ u: I }, { u: H }]));
      this.uniformBuffers.linearQ8_H_I.push(this.makeUniformMixed([{ u: H }, { u: I }]));
      this.uniformBuffers.sizeI.push(this.makeUniformMixed([{ u: I }]));
      this.uniformBuffers.kvCacheStore.push(
        this.makeUniformMixed([{ u: NKH }, { u: HD }, { u: 0 }, { u: MAX_SEQ }])
      );
      // sliding_window: LOCAL (sliding) layers mask to the last
      // `sliding_window` positions; GLOBAL (full-attention) layers attend
      // to everything.
      const slidingWindow = isSwaLayer(il, C) ? C.sliding_window : 0;
      // Attention scaling. Gemma 4 uses 1.0 (q_norm/k_norm per-head RMSNorm
      // already control QK^T magnitude; transformers 5.5.4
      // `Gemma4TextAttention.scaling = 1.0`). Qwen3 keeps the STANDARD
      // 1/sqrt(head_dim) despite also having QK-norm (HF `Qwen3Attention.scaling
      // = head_dim**-0.5`); using 1.0 there makes softmax inputs ~sqrt(128)×
      // too large → attention collapses to one token → degenerate repetition.
      const attnScaling = (C.arch === 'qwen3' || C.arch === 'deepseek-ocr') ? 1.0 / Math.sqrt(HD) : 1.0;
      this.uniformBuffers.attnScore.push(
        this.makeUniformMixed([
          { u: NQH }, { u: NKH }, { u: HD },
          { u: 0 }, { f: attnScaling }, { u: slidingWindow },
        ])
      );
      this.uniformBuffers.attnOutput.push(
        this.makeUniformMixed([{ u: NQH }, { u: NKH }, { u: HD }, { u: 0 }])
      );
      // apply_divisor=1 for GLOBAL layers (rope_freqs table encodes partial_rotary_factor=0.25
      // as divisor: 1.0 for indices 0..63, 1e30 for 64..255 → annihilates rotation on trailing
      // pairs). LOCAL layers use plain 1/pow(theta, ...) schedule with no divisor.
      // Qwen3 uses plain RoPE (theta 1e6, no divisor table) on every layer.
      // Gemma applies the rope_freqs divisor on GLOBAL (full-attention) layers only.
      const applyDivisor = (C.arch === 'qwen3' || C.arch === 'deepseek-ocr') ? 0 : (isSwaLayer(il, C) ? 0 : 1);
      this.uniformBuffers.ropeQ.push(
        this.makeUniformMixed([{ u: NQH }, { u: HD }, { u: 0 }, { f: theta }])
      );
      this.uniformBuffers.ropeK.push(
        this.makeUniformMixed([{ u: NKH }, { u: HD }, { u: 0 }, { f: theta }])
      );
      this.uniformBuffers.fusedNormRopeQ.push(
        this.makeUniformMixed([{ u: NQH }, { u: HD }, { f: C.rms_norm_eps }, { u: 0 }, { f: theta }, { u: applyDivisor }])
      );
      this.uniformBuffers.fusedNormRopeK.push(
        this.makeUniformMixed([{ u: NKH }, { u: HD }, { f: C.rms_norm_eps }, { u: 0 }, { f: theta }, { u: applyDivisor }])
      );
      this.uniformBuffers.pleStage1.push(
        this.makeUniformMixed([{ u: il }, { u: 0 /* token_id */ }, { u: C.per_layer_input_dim }, { f: C.rms_norm_eps }])
      );
      this.uniformBuffers.pleGeluMulParams.push(
        this.makeUniformMixed([{ u: il * C.per_layer_input_dim }, { u: C.per_layer_input_dim }])
      );
    }

    // MoE uniforms (deepseek-ocr). Shapes are identical for every routed
    // layer, so one set serves all of them.
    if (C.moe) {
      const moe = C.moe;
      const mI = moe.moe_intermediate_size;
      this.uniformBuffers.moeRouterMM = this.makeUniformMixed([{ u: moe.num_experts }, { u: H }]);
      this.uniformBuffers.moeTopkParams = this.makeUniformMixed([
        { u: moe.num_experts }, { u: moe.experts_per_tok },
        { u: moe.norm_topk_prob ? 1 : 0 }, { f: moe.routed_scaling_factor },
      ]);
      // Gate/up: M = mI rows per expert over N = hidden; shared input (normed).
      this.uniformBuffers.moeGateUpMM = this.makeUniformMixed([{ u: mI }, { u: H }, { u: mI }, { u: 0 }]);
      // Down: M = hidden rows per expert over N = mI; per-slot input (ffnMul regions).
      this.uniformBuffers.moeDownMM = this.makeUniformMixed([{ u: H }, { u: mI }, { u: H }, { u: 1 }]);
      this.uniformBuffers.sizeMoeAct = this.makeUniformMixed([{ u: moe.experts_per_tok * mI }]);
      this.uniformBuffers.moeAccumParams = this.makeUniformMixed([{ u: H }, { u: moe.experts_per_tok }]);
    }
  }

  /**
   * Create a GPU storage buffer from a Uint16Array of F16 bit patterns.
   * Shader binds as `array<f16>`. WebGPU requires writeBuffer data length + buffer size
   * to be multiples of 4 bytes, so we pad odd-length arrays (e.g., single-scalar tensors
   * like `layer_output_scale.weight` [1]) by appending a zero u16.
   */
  private createF16Buffer(data: Uint16Array, readable = false): GPUBuffer {
    let toWrite = data;
    if ((data.byteLength & 3) !== 0) {
      toWrite = new Uint16Array((data.byteLength + 3) >>> 1 & ~1);
      toWrite.set(data);
    }
    // COPY_SRC is off by default to keep driver overhead minimal on the
    // hundreds of layer weight buffers; opt in for globals that diagnostics
    // may need to read back (per_layer_model_proj layout probe).
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      | (readable ? GPUBufferUsage.COPY_SRC : 0);
    const buf = this.device.createBuffer({ size: toWrite.byteLength, usage });
    this.wb(buf, 0, toWrite);
    return buf;
  }

  /**
   * Requantize an F32 weight [M, N] (N % 32 == 0) to in-shader Q8_0:
   * symmetric int8 with one f16 scale per 32-element block. Returns a packed
   * int8 buffer (4-per-u32) + an f16 scale buffer, both on GPU. ~2× smaller
   * than the F16 representation.
   */
  private createQ8Buffers(f32: Float32Array, M: number, N: number): { quants: GPUBuffer; scales: GPUBuffer } {
    const BLK = 32;
    const nblocks = N / BLK;
    const quants = new Uint8Array(M * N);
    const scales = new Uint16Array(M * nblocks);
    for (let m = 0; m < M; m++) {
      const rowOff = m * N;
      const sOff = m * nblocks;
      for (let b = 0; b < nblocks; b++) {
        const base = rowOff + b * BLK;
        let amax = 0;
        for (let i = 0; i < BLK; i++) { const v = Math.abs(f32[base + i]); if (v > amax) amax = v; }
        const scale = amax / 127;
        const inv = scale > 0 ? 1 / scale : 0;
        scales[sOff + b] = f32ToF16(scale);
        for (let i = 0; i < BLK; i++) {
          let q = Math.round(f32[base + i] * inv);
          if (q > 127) q = 127; else if (q < -127) q = -127;
          quants[base + i] = q & 0xff;
        }
      }
    }
    const qU32 = new Uint32Array(quants.buffer); // M*N % 4 == 0 (N % 32 == 0)
    const quantBuf = this.device.createBuffer({
      size: qU32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.wb(quantBuf, 0, qU32);
    const scaleBuf = this.createF16Buffer(scales);
    return { quants: quantBuf, scales: scaleBuf };
  }

  /** Build a Q8 matmul bind group (matmulQ8 layout: input, quants, scales, output, params). */
  private mmQ8(input: GPUBuffer, layer: Record<string, GPUBuffer>, key: string, output: GPUBuffer, params: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.pipelines.matmulQ8.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: layer[key] } },
        { binding: 2, resource: { buffer: layer[`${key}__s`] } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
  }

  /**
   * Requantize an F32 weight [M, N] (N % 256 == 0) to in-shader 4-bit block-affine:
   * GGUF Q4_K's structure (8 sub-blocks of 32 per 256-super-block, each an
   * independent 4-bit affine grid) but with each sub-block's scale+min kept as f16
   * rather than 6-bit-packed via a super-scale. Strictly higher precision than GGUF
   * Q4_K, so a Q4_K source round-trips near-losslessly. Returns a packed 4-bit
   * buffer (8-per-u32) + an f16 meta buffer (16 per super-block: 8 scales, 8 mins),
   * both on GPU. ~4× smaller than F16 (5 bits/weight incl. the f16 meta).
   */
  private createQ4KBuffers(f32: Float32Array, M: number, N: number): { quants: GPUBuffer; meta: GPUBuffer } {
    const SUB = 32;
    const SUPER = 256;
    const nsblk = N / SUPER;          // super-blocks per row
    const nibbles = new Uint8Array(M * N); // 0..15 per element (packed below)
    const meta = new Uint16Array(M * nsblk * 16); // 8 scales + 8 mins per super-block
    for (let m = 0; m < M; m++) {
      const rowOff = m * N;
      const rowMeta = m * nsblk * 16;
      for (let sb = 0; sb < nsblk; sb++) {
        const sbBase = rowOff + sb * SUPER;
        const mBase = rowMeta + sb * 16;
        for (let sub = 0; sub < 8; sub++) {
          const base = sbBase + sub * SUB;
          let mn = Infinity, mx = -Infinity;
          for (let i = 0; i < SUB; i++) { const v = f32[base + i]; if (v < mn) mn = v; if (v > mx) mx = v; }
          const scale = (mx - mn) / 15;
          const inv = scale > 0 ? 1 / scale : 0;
          meta[mBase + sub] = f32ToF16(scale);
          meta[mBase + 8 + sub] = f32ToF16(mn);
          for (let i = 0; i < SUB; i++) {
            let q = Math.round((f32[base + i] - mn) * inv);
            if (q > 15) q = 15; else if (q < 0) q = 0;
            nibbles[base + i] = q;
          }
        }
      }
    }
    // Pack 8 nibbles per u32 (little-endian nibble order, matching the shader).
    const qU32 = new Uint32Array((M * N) / 8);
    for (let i = 0; i < M * N; i++) {
      qU32[i >> 3] |= (nibbles[i] & 0xf) << ((i & 7) * 4);
    }
    const quantBuf = this.device.createBuffer({
      size: qU32.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.wb(quantBuf, 0, qU32);
    const metaBuf = this.createF16Buffer(meta);
    return { quants: quantBuf, meta: metaBuf };
  }

  /** Build an in-shader-quant matmul bind group. matmulQ8 and matmulQ4K share the
   *  layout (input, quants, scale-meta, output, params); pick the pipeline by mode. */
  private mmQuant(input: GPUBuffer, layer: Record<string, GPUBuffer>, key: string, output: GPUBuffer, params: GPUBuffer): GPUBindGroup {
    const pipe = this.weightQuant === 'q4k' ? this.pipelines.matmulQ4KMr : this.pipelines.matmulQ8Mr;
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: layer[key] } },
        { binding: 2, resource: { buffer: layer[`${key}__s`] } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
  }

  /** In-shader-quant matmul bind group for an EXPLICIT pipeline (deepseek-ocr
   *  mixes q4k and q8 storage within one layer — the mode-derived mmQuant
   *  can't express that). Same 5-entry layout as mmQuant. */
  private mmWith(pipe: GPUComputePipeline, input: GPUBuffer, layer: Record<string, GPUBuffer>, key: string, output: GPUBuffer, params: GPUBuffer): GPUBindGroup {
    return this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: layer[key] } },
        { binding: 2, resource: { buffer: layer[`${key}__s`] } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
  }

  /** Dequantize any GGUF tensor to F16 bit patterns. One-time CPU cost at upload. */
  private tensorToF16(parser: GGUFParser, tensor: GGUFTensor, dataOffset: number): Uint16Array {
    // F16 source can be used directly without the F32 round-trip.
    if (tensor.type === 1) {
      const absOff = dataOffset + Number(tensor.offset);
      const numElems = Number(tensor.dims.reduce((a, b) => a * b, 1n));
      return new Uint16Array(parser.buffer.slice(absOff, absOff + numElems * 2));
    }
    const localOffset = dataOffset + Number(tensor.offset);
    const count = Number(tensor.dims.reduce((a, b) => a * b, 1n));
    if (count > F16_CHUNK_THRESHOLD) {
      return this.dequantToF16Chunked(parser, tensor, localOffset, count);
    }
    return f32ToF16Array(parser.getTensorData(tensor, dataOffset));
  }

  /**
   * Decode a quantized tensor to F16 in element-chunks, avoiding a single giant
   * F32 intermediate. The 8B token_embd (622M elems) would need a 2.49 GB
   * Float32Array in one shot — past V8's allocation ceiling. We decode ~16M
   * elems at a time into a small scratch and convert into the (single, 1.24 GB)
   * F16 output. `byteOffset` is an absolute offset into `parser`'s buffer; it
   * must sit on a super-block boundary (tensor data always does). K-quant and
   * Q8_0 types only — other types never reach the size that needs chunking.
   */
  private dequantToF16Chunked(parser: GGUFParser, tensor: GGUFTensor, byteOffset: number, count: number): Uint16Array {
    let elemsPerBlk: number, bytesPerBlk: number;
    switch (tensor.type) {
      case 12: elemsPerBlk = 256; bytesPerBlk = 144; break; // Q4_K
      case 13: elemsPerBlk = 256; bytesPerBlk = 176; break; // Q5_K
      case 14: elemsPerBlk = 256; bytesPerBlk = 210; break; // Q6_K
      case 8:  elemsPerBlk = 32;  bytesPerBlk = 34;  break; // Q8_0
      case 6:  elemsPerBlk = 32;  bytesPerBlk = 22;  break; // Q5_0
      default: return f32ToF16Array(parser.getTensorData({ ...tensor, offset: BigInt(byteOffset) }, 0));
    }
    const out = new Uint16Array(count);
    const chunkElems = Math.max(elemsPerBlk, Math.floor((1 << 24) / elemsPerBlk) * elemsPerBlk);
    let elem = 0, off = byteOffset;
    while (elem < count) {
      const n = Math.min(chunkElems, count - elem);
      let f32: Float32Array;
      switch (tensor.type) {
        case 12: f32 = parser.dequantizeQ4_K(off, n); break;
        case 13: f32 = parser.dequantizeQ5_K(off, n); break;
        case 14: f32 = parser.dequantizeQ6_K(off, n); break;
        case 6:  f32 = parser.dequantizeQ5_0(off, n); break;
        default: f32 = parser.dequantizeQ8_0(off, n); break;
      }
      for (let i = 0; i < n; i++) out[elem + i] = f32ToF16(f32[i]);
      elem += n;
      off += (n / elemsPerBlk) * bytesPerBlk;
    }
    return out;
  }

  private async uploadWeightsFromBuffer(parser: GGUFParser, gguf: { tensors: GGUFTensor[]; dataOffset: number }): Promise<void> {
    if (this.config.arch === 'deepseek-ocr') {
      throw new Error('deepseek-ocr needs the streaming (Range) load path — serve the GGUF from a server that honors Range requests');
    }
    const tensors = gguf.tensors;
    const dataOffset = gguf.dataOffset;

    this.modelBuffers = {
      tokenEmbed: null,
      layers: [],
      finalNorm: null,
      globals: {},
      perLayerEmbeddings: [],
    };

    const uploadToF16 = (t: GGUFTensor, readable = false): GPUBuffer =>
      this.createF16Buffer(this.tensorToF16(parser, t, dataOffset), readable);

    // Weight-layout probes: every uploaded weight opts into COPY_SRC so we
    // can read back slices for bytewise comparison vs GGUF source. Driver
    // overhead is paid once at upload; no hot-path cost. Strip `readable`
    // if the diagnostic capability isn't needed and bench shows regression.
    const embedTensor = tensors.find(t => t.name === 'token_embd.weight');
    if (embedTensor) this.modelBuffers.tokenEmbed = uploadToF16(embedTensor, true);

    const finalNormTensor = tensors.find(t => t.name === 'output_norm.weight');
    if (finalNormTensor) this.modelBuffers.finalNorm = uploadToF16(finalNormTensor, true);

    // Global tensors except per_layer_token_embd (which is too big for one buffer — split below).
    for (const name of [...GLOBAL_F32_TENSORS]) {
      const tensor = tensors.find(t => t.name === name + '.weight');
      if (tensor) this.modelBuffers.globals[name] = uploadToF16(tensor, true);
    }

    // Per-layer embedding table (Gemma 4 PLE). Split along layer dim to stay under 2 GB/buffer.
    await this.uploadPerLayerEmbeddingsFromBuffer(parser, tensors, dataOffset);

    for (let i = 0; i < this.config.num_layers; i++) {
      const prefix = `blk.${i}.`;
      const layer: Record<string, GPUBuffer> = {};
      for (const key of [...LAYER_F32_WEIGHT_NAMES, ...LAYER_MATMUL_WEIGHT_NAMES]) {
        const tensor = tensors.find(t => t.name === prefix + key + '.weight');
        if (tensor) layer[key] = uploadToF16(tensor, true);
      }
      this.modelBuffers.layers.push(layer);
    }
  }

  /**
   * Stream-decode the Q5_K per_layer_token_embd tensor into 35 per-layer F16 GPU buffers.
   *
   * Invariant exploited: each Q5_K super-block is 256 elements = exactly one (vocab, layer)
   * slice since per_layer_input_dim = 256. Super-block index s maps to vocab = s / num_layers,
   * layer = s % num_layers (because GGUF inner dim is num_layers*per_layer_dim, outer is vocab).
   *
   * The decoder consumes an `AsyncIterable<Uint8Array>` of network chunks rather than
   * a pre-buffered 1.6 GB slab. Fetch and decode overlap naturally: while JS runs a
   * decode burst, the browser's network stack continues filling its internal buffer;
   * when JS awaits the next chunk, it drains accumulated bytes in one gulp. Peak JS
   * memory is `processingChunkBytes` (~3.15 MB accumulator) plus staging / encoding
   * buffers (~9 MB across 35 layers) — orders of magnitude below the old ~1.6 GB.
   *
   * The full-buffer fallback path synthesizes a single-chunk async iterable so both
   * paths share this one decoder.
   */
  private async streamDecodePleQ5_K(
    chunks: AsyncIterable<Uint8Array>,
    totalBytes: number,
  ): Promise<void> {
    const numLayers = this.config.num_layers;
    const perLayerDim = this.config.per_layer_input_dim;
    const vocab = this.config.vocab_size;
    const BLOCK_BYTES = 176;
    const BLOCK_ELEMS = 256;
    if (perLayerDim !== BLOCK_ELEMS) {
      throw new Error(`PLE streaming assumes per_layer_input_dim == 256 (got ${perLayerDim})`);
    }

    // Pre-allocate 35 GPU buffers of [vocab × 256] F16 each.
    // COPY_SRC supports the captureEmbedSliceAll diagnostic; negligible runtime cost.
    const sliceBytes = vocab * perLayerDim * 2;
    for (let L = 0; L < numLayers; L++) {
      const buf = this.device.createBuffer({
        size: (sliceBytes + 3) & ~3,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      });
      this.modelBuffers.perLayerEmbeddings.push(buf);
    }

    // Per-layer f32 staging; bulk f32→f16 once per (chunk, layer) keeps native
    // Float16Array fast-paths hot.
    const CHUNK = 512; // vocab entries per processing chunk
    const f32StagingPerLayer: Float32Array[] = [];
    for (let L = 0; L < numLayers; L++) {
      f32StagingPerLayer.push(new Float32Array(CHUNK * perLayerDim));
    }

    // Flat accumulator sized to exactly one processing chunk. We drain when it
    // fills. `bytesPerVocab` = 35 × 176 = 6160 bytes; CHUNK × bytesPerVocab ≈ 3.15 MB.
    const bytesPerVocab = numLayers * BLOCK_BYTES;
    const processingChunkBytes = CHUNK * bytesPerVocab;
    const accumBuf = new Uint8Array(processingChunkBytes);
    let accumLen = 0;
    let vocabProcessed = 0;
    let bytesReceived = 0;
    const totalVocab = vocab;

    const drainChunk = (vChunkSize: number): void => {
      const parser = new GGUFParser(accumBuf.subarray(0, vChunkSize * bytesPerVocab));
      for (let cv = 0; cv < vChunkSize; cv++) {
        const stageOff = cv * perLayerDim;
        for (let L = 0; L < numLayers; L++) {
          // Super-block layout within the accumulator: (cv * numLayers + L) * BLOCK_BYTES.
          const blockByteOff = (cv * numLayers + L) * BLOCK_BYTES;
          parser.decodeQ5_KBlock(blockByteOff, f32StagingPerLayer[L], stageOff);
        }
      }
      for (let L = 0; L < numLayers; L++) {
        const f32Chunk = vChunkSize === CHUNK
          ? f32StagingPerLayer[L]
          : f32StagingPerLayer[L].subarray(0, vChunkSize * perLayerDim);
        const f16Chunk = f32ToF16Array(f32Chunk);
        this.device.queue.writeBuffer(
          this.modelBuffers.perLayerEmbeddings[L],
          vocabProcessed * perLayerDim * 2,
          f16Chunk.buffer,
          0,
          vChunkSize * perLayerDim * 2,
        );
      }
      vocabProcessed += vChunkSize;
      accumLen = 0;
      const pct = (vocabProcessed / totalVocab * 100).toFixed(1);
      this.reportProgress(
        vocabProcessed,
        totalVocab,
        `PLE pipeline: ${pct}% (vocab ${vocabProcessed}/${totalVocab})`,
      );
    };

    for await (const netChunk of chunks) {
      bytesReceived += netChunk.byteLength;
      let netOffset = 0;
      while (netOffset < netChunk.byteLength) {
        const space = processingChunkBytes - accumLen;
        const copy = Math.min(space, netChunk.byteLength - netOffset);
        accumBuf.set(netChunk.subarray(netOffset, netOffset + copy), accumLen);
        accumLen += copy;
        netOffset += copy;
        if (accumLen === processingChunkBytes) {
          drainChunk(CHUNK);
        }
      }
    }

    if (bytesReceived !== totalBytes) {
      throw new Error(`PLE streaming: received ${bytesReceived} bytes but expected ${totalBytes}`);
    }

    // Final partial chunk — the last processing chunk may carry fewer than CHUNK vocab
    // entries when vocab is not a multiple of CHUNK (Gemma 4 E2B: 262144 is a multiple,
    // but don't assume).
    if (accumLen > 0) {
      const remainingVocab = totalVocab - vocabProcessed;
      if (remainingVocab * bytesPerVocab !== accumLen) {
        throw new Error(
          `PLE streaming: accumulator has ${accumLen} bytes but ${remainingVocab} vocab entries remain (expected ${remainingVocab * bytesPerVocab})`,
        );
      }
      drainChunk(remainingVocab);
    }

    if (vocabProcessed !== totalVocab) {
      throw new Error(`PLE streaming: processed ${vocabProcessed} vocab entries, expected ${totalVocab}`);
    }
  }

  /**
   * Non-streaming variant — slice the per_layer_token_embd tensor bytes out of the
   * fully-loaded GGUF buffer and feed the streaming decoder a single-chunk iterable.
   * Used only by uploadWeightsFromBuffer (server without Range support — rare).
   */
  private async uploadPerLayerEmbeddingsFromBuffer(
    parser: GGUFParser,
    tensors: GGUFTensor[],
    dataOffset: number,
  ): Promise<void> {
    const tensor = tensors.find(t => t.name === 'per_layer_token_embd.weight');
    if (!tensor) return;
    const abs = dataOffset + Number(tensor.offset);
    const size = tensorByteSize(tensor);
    const bytes = new Uint8Array(parser.buffer, abs, size);
    async function* single(): AsyncGenerator<Uint8Array, void, void> {
      yield bytes;
    }
    await this.streamDecodePleQ5_K(single(), size);
  }

  private async uploadWeightsStreaming(
    modelFile: string,
    tensors: GGUFTensor[],
    dataOffset: number,
    signal?: AbortSignal,
  ): Promise<void> {
    // deepseek-ocr P0 scope: the MoE expert packs are only wired for the
    // in-shader-q4k storage mode (gate/up exps q4k; down projections q8 —
    // their N isn't a multiple of the 256-element q4k super-block).
    if (this.config.arch === 'deepseek-ocr' && this.weightQuant !== 'q4k') {
      throw new Error("deepseek-ocr requires weightQuant: 'q4k' (MoE expert packs are q4k/q8 in-shader only)");
    }
    this.modelBuffers = {
      tokenEmbed: null,
      layers: [],
      finalNorm: null,
      globals: {},
      perLayerEmbeddings: [],
    };

    // RangedReader handles the known failure modes: HF xet-bridge
    // connection-resets (retry with backoff) and silent TCP-zombie
    // stalls (no-progress timeout + resume-from-offset). The latter
    // was the "stuck at 8.2%" gremlin on the 1.6 GB PLE fetch —
    // arrayBuffer() would hang forever with no signal to act on.
    //
    // Retry events surface through the engine's onProgress callback
    // with kind='retrying' so UI surfaces can render "connection
    // stalled, retrying…" instead of an eerily quiet progress bar.
    // The onEvent closure reads `totalUploaded` / `totalWeightBytes`
    // via the enclosing scope; they're declared below but the
    // callback only fires during later fetch events, by which point
    // the bindings exist.
    const rangedReader = new RangedReader(modelFile, {
      noProgressTimeoutMs: 30_000,
      maxAttempts: 4,
      signal,
      onEvent: (ev) => {
        if (ev.type === 'retry') {
          console.warn(
            `[gemma4-webgpu] Range reader retry #${ev.attempt} after ${ev.bytesReadBeforeRetry} bytes: ${ev.reason}`,
          );
          this.reportProgress(
            totalUploaded,
            totalWeightBytes,
            `Connection stalled, retrying (attempt ${ev.attempt})…`,
            'retrying',
          );
        }
      },
    });
    const fetchRange = (start: number, size: number): Promise<Uint8Array> =>
      rangedReader.readAll(start, size);

    /**
     * Convert a byte slice containing a single GGUF tensor into a Uint16Array of F16
     * bit patterns. Source dequant happens via a temp parser; result is ready to
     * upload to a WebGPU storage buffer bound as array<f16>.
     */
    const tensorBytesToF16 = (bytes: Uint8Array, localOffset: number, tensor: GGUFTensor): Uint16Array => {
      // F16 source — no conversion needed.
      if (tensor.type === 1) {
        const numElements = Number(tensor.dims.reduce((a, b) => a * b, 1n));
        return new Uint16Array(bytes.buffer.slice(bytes.byteOffset + localOffset, bytes.byteOffset + localOffset + numElements * 2));
      }
      // Everything else: CPU-dequant to F32, convert to F16. Large tensors (8B
      // token_embd / lm_head) decode in chunks to dodge a >2 GB F32 intermediate.
      const tempParser = new GGUFParser(bytes);
      const count = Number(tensor.dims.reduce((a, b) => a * b, 1n));
      if (count > F16_CHUNK_THRESHOLD) {
        return this.dequantToF16Chunked(tempParser, tensor, localOffset, count);
      }
      const f32 = tempParser.getTensorData(
        { ...tensor, offset: BigInt(localOffset) },
        0,
      );
      return f32ToF16Array(f32);
    };

    const uploadTensor = (bytes: Uint8Array, localOffset: number, tensor: GGUFTensor, readable = false): { buf: GPUBuffer; byteLength: number } => {
      const f16 = tensorBytesToF16(bytes, localOffset, tensor);
      const buf = this.createF16Buffer(f16, readable);
      return { buf, byteLength: f16.byteLength };
    };

    // Q8 path: dequant source → F32 → requantize to in-shader Q8_0. dims[0]=N
    // (contraction/input) in GGUF's reversed layout; M = the remaining dims'
    // product — dims[1] for 2-D weights, rows x experts for 3-D MoE expert
    // packs ([N, rows, experts]), which lays every expert's rows out
    // contiguously exactly as the expert kernels index them.
    const uploadTensorQ8 = (bytes: Uint8Array, localOffset: number, tensor: GGUFTensor): { quants: GPUBuffer; scales: GPUBuffer } => {
      const tp = new GGUFParser(bytes);
      const f32 = tp.getTensorData({ ...tensor, offset: BigInt(localOffset) }, 0);
      const N = Number(tensor.dims[0]);
      const M = f32.length / N;
      return this.createQ8Buffers(f32, M, N);
    };

    // Q4_K path: dequant source → F32 → requantize to in-shader 4-bit block-affine.
    // Same M/N convention as the Q8 path (3-D expert packs supported).
    const uploadTensorQ4K = (bytes: Uint8Array, localOffset: number, tensor: GGUFTensor): { quants: GPUBuffer; meta: GPUBuffer } => {
      const tp = new GGUFParser(bytes);
      const f32 = tp.getTensorData({ ...tensor, offset: BigInt(localOffset) }, 0);
      const N = Number(tensor.dims[0]);
      const M = f32.length / N;
      return this.createQ4KBuffers(f32, M, N);
    };

    let totalUploaded = 0;
    const totalWeightBytes = tensors.reduce((sum, t) => sum + tensorByteSize(t), 0);

    // Phase accumulators for this._loadTimings. Each layer worker contributes
    // its fetch/process durations and byte size to these running sums; order of
    // contribution doesn't matter for the aggregates.
    let layersFetchAcc = 0;
    let layersProcessAcc = 0;
    let layerBytesAcc = 0;

    /** Fetch one tensor worth of bytes by name, upload, return destination. */
    const streamOneTensor = async (name: string, readable = false): Promise<GPUBuffer | null> => {
      const tensor = tensors.find(t => t.name === name + '.weight');
      if (!tensor) return null;
      const fileOffset = dataOffset + Number(tensor.offset);
      const size = tensorByteSize(tensor);
      const bytes = await fetchRange(fileOffset, size);
      const result = uploadTensor(bytes, 0, tensor, readable);
      totalUploaded += size;
      return result.buf;
    };

    // Upload main embedding (F16 on GPU). Marked readable for the
    // tied-LM-head diagnostic probe.
    this.modelBuffers.tokenEmbed = await streamOneTensor('token_embd', true);
    this.reportProgress(totalUploaded, totalWeightBytes, 'Streaming weights to GPU...');

    // Upload final norm. Marked readable for the finalNorm diagnostic probe.
    this.modelBuffers.finalNorm = await streamOneTensor('output_norm', true);

    // Untied LM head (Qwen3 4B/8B ship a separate `output.weight`; smaller
    // models tie embeddings). When present, used as the LM-head weight instead
    // of token_embd.
    if (this.config.lm_head_tensor) {
      const lmHeadBuf = await streamOneTensor(this.config.lm_head_tensor, true);
      if (lmHeadBuf) this.modelBuffers.globals['lm_head'] = lmHeadBuf;
    }

    // Small global tensors (per_layer_model_proj, per_layer_proj_norm, rope_freqs).
    // Uploaded with COPY_SRC to support weight-layout diagnostic probes.
    for (const name of GLOBAL_F32_TENSORS) {
      const buf = await streamOneTensor(name, true);
      if (buf) {
        this.modelBuffers.globals[name] = buf;
        this.reportProgress(totalUploaded, totalWeightBytes, `Loaded ${name}`);
      }
    }

    // per_layer_token_embd is Q5_K with ~1.6 GB of source bytes. A full dequant to f32
    // would need 9.4 GB — exceeds V8's per-tab heap. Stream the bytes from the network
    // directly into the super-block decoder; fetch and decode overlap, and peak memory
    // stays at ~3 MB of accumulator + ~9 MB of per-layer staging. The old path held all
    // 1.6 GB in JS memory while also being the exact chokepoint of the stall-at-8.2%
    // gremlin — RangedReader.stream() resolves both.
    const ple = tensors.find(t => t.name === 'per_layer_token_embd.weight');
    if (ple) {
      const fileOffset = dataOffset + Number(ple.offset);
      const size = tensorByteSize(ple);
      this.reportProgress(totalUploaded, totalWeightBytes, `Streaming per_layer_token_embd (~${(size / 1e9).toFixed(2)} GB Q5_K)…`);
      if (this._loadTimings) {
        this._loadTimings.pleBytes = size;
      }
      const plePipelineStart = performance.now();
      await this.streamDecodePleQ5_K(rangedReader.stream(fileOffset, size), size);
      if (this._loadTimings) {
        this._loadTimings.phases.plePipeline_ms = performance.now() - plePipelineStart;
      }
      totalUploaded += size;
    }

    // Upload layers concurrently via Range requests. Each worker fetches one
    // layer's tensor bytes, then synchronously dequants + uploads to GPU on the
    // main thread; while one worker is in the CPU-bound upload phase, the other
    // workers' HTTP requests continue filling their browser-side buffers, so
    // network and CPU work naturally pipeline.
    //
    // Order preservation: modelBuffers.layers[] and perLayer[] are pre-sized and
    // populated by index, so finish-order doesn't matter — modelBuffers.layers[i]
    // is always layer i even if i=7 finished before i=3.
    const numLayers = this.config.num_layers;
    const LAYER_CONCURRENCY = 4;
    this.modelBuffers.layers = new Array(numLayers);
    const perLayerByIdx: Array<NonNullable<LoadTimings['perLayer']>[number] | null> = new Array(numLayers).fill(null);
    let completedLayers = 0;

    const isDsOcr = this.config.arch === 'deepseek-ocr';
    /**
     * Find a layer tensor by engine key. deepseek-ocr MoE layers have no
     * dense `ffn_gate/up/down` — the pre-fused shared-expert FFN
     * (`ffn_*_shexp`) aliases onto those keys so the standard dense-FFN path
     * computes shared_experts(x) with zero forward-pass changes.
     */
    const findLayerTensor = (layerTensors: GGUFTensor[], prefix: string, key: string): GGUFTensor | undefined => {
      const t = layerTensors.find(x => x.name === prefix + key + '.weight');
      if (t || !isDsOcr) return t;
      if (key === 'ffn_gate' || key === 'ffn_up' || key === 'ffn_down') {
        return layerTensors.find(x => x.name === prefix + key + '_shexp.weight');
      }
      return undefined;
    };

    const uploadOneLayer = async (i: number): Promise<void> => {
      const prefix = `blk.${i}.`;
      const layerTensors = tensors.filter(t => t.name.startsWith(prefix));
      if (layerTensors.length === 0) return;

      let minOffset = Infinity, maxEnd = 0;
      for (const t of layerTensors) {
        const off = Number(t.offset);
        const end = off + tensorByteSize(t);
        if (off < minOffset) minOffset = off;
        if (end > maxEnd) maxEnd = end;
      }

      const layerStart = dataOffset + minOffset;
      const layerSize = maxEnd - minOffset;
      const fetchStart = performance.now();
      const layerBytes = await fetchRange(layerStart, layerSize);
      const fetchMs = performance.now() - fetchStart;

      const processStart = performance.now();
      const layer: Record<string, GPUBuffer> = {};
      for (const key of LAYER_F32_WEIGHT_NAMES) {
        const tensor = layerTensors.find(t => t.name === prefix + key + '.weight');
        if (tensor) {
          const localOffset = Number(tensor.offset) - minOffset;
          const result = uploadTensor(layerBytes, localOffset, tensor, true);
          layer[key] = result.buf;
        }
      }
      for (const key of LAYER_MATMUL_WEIGHT_NAMES) {
        const tensor = findLayerTensor(layerTensors, prefix, key);
        if (tensor) {
          const localOffset = Number(tensor.offset) - minOffset;
          // deepseek-ocr down projections go q8: the dense layer-0 down
          // (N=6848) and the routed-expert down (N=896) aren't multiples of
          // the 256-elem q4k super-block; shexp down (N=1792) would be, but
          // shares the q8 path for uniformity — and q8 is the higher-fidelity
          // round-trip for these Q5_0/Q8_0-source tensors anyway.
          const wantQ8 = this.weightQuant === 'q8' || (isDsOcr && key === 'ffn_down');
          if (wantQ8) {
            const { quants, scales } = uploadTensorQ8(layerBytes, localOffset, tensor);
            layer[key] = quants;
            layer[`${key}__s`] = scales;
          } else if (this.weightQuant === 'q4k') {
            const { quants, meta } = uploadTensorQ4K(layerBytes, localOffset, tensor);
            layer[key] = quants;
            layer[`${key}__s`] = meta;
          } else {
            layer[key] = uploadTensor(layerBytes, localOffset, tensor, true).buf;
          }
        }
      }
      // deepseek-ocr MoE tensors: router (tiny, F16), routed expert packs
      // (gate/up q4k over N=hidden; down q8 over N=moe_I — see wantQ8 note).
      if (isDsOcr) {
        const router = layerTensors.find(x => x.name === prefix + 'ffn_gate_inp.weight');
        if (router) {
          layer['ffn_gate_inp'] = uploadTensor(layerBytes, Number(router.offset) - minOffset, router, true).buf;
        }
        for (const key of ['ffn_gate_exps', 'ffn_up_exps']) {
          const t = layerTensors.find(x => x.name === prefix + key + '.weight');
          if (t) {
            const { quants, meta } = uploadTensorQ4K(layerBytes, Number(t.offset) - minOffset, t);
            layer[key] = quants;
            layer[`${key}__s`] = meta;
          }
        }
        const downExps = layerTensors.find(x => x.name === prefix + 'ffn_down_exps.weight');
        if (downExps) {
          const { quants, scales } = uploadTensorQ8(layerBytes, Number(downExps.offset) - minOffset, downExps);
          layer['ffn_down_exps'] = quants;
          layer['ffn_down_exps__s'] = scales;
        }
      }
      this.modelBuffers.layers[i] = layer;
      const processMs = performance.now() - processStart;

      // Accumulators — `+=` is atomic at the JS event-loop level, so concurrent
      // workers don't race here.
      layersFetchAcc += fetchMs;
      layersProcessAcc += processMs;
      layerBytesAcc += layerSize;
      perLayerByIdx[i] = { layerIdx: i, fetchMs, processMs, bytes: layerSize };

      totalUploaded += layerSize;
      completedLayers++;
      this.reportProgress(totalUploaded, totalWeightBytes, `Layers ${completedLayers}/${numLayers}`);
    };

    // Bounded-concurrency scheduler: keep up to LAYER_CONCURRENCY promises in
    // flight, awaiting the first to complete before launching the next.
    const inFlight = new Set<Promise<void>>();
    for (let i = 0; i < numLayers; i++) {
      const p = uploadOneLayer(i).finally(() => inFlight.delete(p));
      inFlight.add(p);
      if (inFlight.size >= LAYER_CONCURRENCY) {
        await Promise.race(inFlight);
      }
    }
    await Promise.all(inFlight);

    if (this._loadTimings) {
      // perLayer array is populated by index so it's still in layer order even
      // though workers completed out of order.
      this._loadTimings.phases.layersFetch_ms = layersFetchAcc;
      this._loadTimings.phases.layersProcess_ms = layersProcessAcc;
      this._loadTimings.phases.layersTotal_ms = layersFetchAcc + layersProcessAcc;
      this._loadTimings.perLayer = perLayerByIdx.filter((x): x is NonNullable<typeof x> => x !== null);
      this._loadTimings.layerBytesTotal = layerBytesAcc;
    }
  }

  private createWorkBuffers(): void {
    const C = this.config;
    const H = C.hidden_size;
    const Q = C.q_dim;
    const KV = C.kv_dim;
    const I = C.intermediate_size;
    const MAX_SEQ = C.context_length;
    const V = C.vocab_size;
    const NKH = C.num_kv_heads;

    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const SD = S | GPUBufferUsage.COPY_DST;
    this.workBuffers = {
      hidden: this.device.createBuffer({ size: H * 4, usage: SD }),
      // Sized for the max of any capture target: hidden [H], normed [H], pleInputs [L*P].
      // Diagnostic readback only; never touched on the hot path.
      hiddenReadback: this.device.createBuffer({
        size: Math.max(H, C.num_layers * C.per_layer_input_dim, 1) * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      }),
      residual: this.device.createBuffer({ size: H * 4, usage: S }),
      normed: this.device.createBuffer({ size: H * 4, usage: S }),
      q: this.device.createBuffer({ size: Q * 4, usage: S }),
      k: this.device.createBuffer({ size: KV * 4, usage: S }),
      v: this.device.createBuffer({ size: KV * 4, usage: S }),
      attnOut: this.device.createBuffer({ size: Q * 4, usage: S }),
      attnProj: this.device.createBuffer({ size: H * 4, usage: S }),
      postAttnNormed: this.device.createBuffer({ size: H * 4, usage: S }),
      attnScores: this.device.createBuffer({ size: C.num_q_heads * MAX_SEQ * 4, usage: S }),
      ffnGate: this.device.createBuffer({ size: I * 4, usage: S }),
      ffnUp: this.device.createBuffer({ size: I * 4, usage: S }),
      ffnMul: this.device.createBuffer({ size: I * 4, usage: S }),
      ffnDown: this.device.createBuffer({ size: H * 4, usage: S }),
      postFfnNormed: this.device.createBuffer({ size: H * 4, usage: S }),
      logits: this.device.createBuffer({ size: V * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      logitsReadback: this.device.createBuffer({ size: V * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      argmaxResult: this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      argmaxReadbackPool: [
        this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
        this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      ],
      topk256Result: this.device.createBuffer({ size: 256 * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      topk256Readback: this.device.createBuffer({ size: 256 * 2 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      // Per-layer embedding (PLE) work buffers.
      plePmProjected: this.device.createBuffer({ size: Math.max(4, C.num_layers * C.per_layer_input_dim * 4), usage: S }),
      pleInputs: this.device.createBuffer({ size: Math.max(4, C.num_layers * C.per_layer_input_dim * 4), usage: S }),
      pleGate: this.device.createBuffer({ size: Math.max(4, C.per_layer_input_dim * 4), usage: S }),
      pleGated: this.device.createBuffer({ size: Math.max(4, C.per_layer_input_dim * 4), usage: S }),
      pleProjOut: this.device.createBuffer({ size: H * 4, usage: S }),
      plePostNormed: this.device.createBuffer({ size: H * 4, usage: S }),
    };

    // MoE work buffers (deepseek-ocr). The routed experts reuse ffnGate /
    // ffnUp / ffnMul (sized for I_max = 6848 >= experts_per_tok x moe_I = 5376)
    // — only the router logits, the top-k selection, and the per-slot
    // down-proj outputs need dedicated buffers.
    if (C.moe) {
      this.workBuffers.moeRouterLogits = this.device.createBuffer({
        size: Math.max(C.moe.num_experts, 4) * 4, usage: S,
      });
      this.workBuffers.moeSel = this.device.createBuffer({
        size: C.moe.experts_per_tok * 2 * 4, usage: S,
      });
      this.workBuffers.moeSlots = this.device.createBuffer({
        size: C.moe.experts_per_tok * H * 4, usage: S,
      });
    }

    // Per-layer KV cache sizes. On Gemma 4 LOCAL layers use HD=256, GLOBAL layers use HD=512.
    // Sizing per layer (instead of uniformly at max) saves a few tens of MB at ctx=2048.
    this.kvCaches = [];
    for (let i = 0; i < C.num_layers; i++) {
      const layerHD = getHeadDim(i, C);
      const kvSize = MAX_SEQ * NKH * layerHD * 4;
      this.kvCaches.push({
        k: this.device.createBuffer({ size: kvSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }),
        v: this.device.createBuffer({ size: kvSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }),
      });
    }
  }

  private createBindGroups(): void {
    // PLE machinery exists only on the Gemma 4 path.
    const hasPle = this.config.arch === undefined || this.config.arch === 'gemma4';
    // Per-layer embedding bind groups — built first so they can land
    // directly in the BindGroupCache object literal below. Qwen3 has no PLE,
    // and the per_layer_model_proj / per_layer_proj_norm / perLayerEmbeddings
    // buffers are never loaded for it — so skip these entirely.
    const plePmProjMatmul = !hasPle ? undefined : this.device.createBindGroup({
      layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.workBuffers.hidden } },
        { binding: 1, resource: { buffer: this.modelBuffers.globals.per_layer_model_proj } },
        { binding: 2, resource: { buffer: this.workBuffers.plePmProjected } },
        { binding: 3, resource: { buffer: this.uniformBuffers.plePmProjMM } },
      ],
    });
    const pleStage1Fuse: GPUBindGroup[] = [];
    if (hasPle) {
      for (let i = 0; i < this.config.num_layers; i++) {
        pleStage1Fuse.push(this.device.createBindGroup({
          layout: this.pipelines.pleStage1Fuse.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.plePmProjected } },
            { binding: 1, resource: { buffer: this.modelBuffers.globals.per_layer_proj_norm } },
            { binding: 2, resource: { buffer: this.modelBuffers.perLayerEmbeddings[i] } },
            { binding: 3, resource: { buffer: this.workBuffers.pleInputs } },
            { binding: 4, resource: { buffer: this.uniformBuffers.pleStage1[i] } },
          ],
        }));
      }
    }

    const bgc: BindGroupCache = {
      embeddingLookup: this.device.createBindGroup({
        layout: this.pipelines.embeddingLookup.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.modelBuffers.tokenEmbed! } },
          { binding: 1, resource: { buffer: this.workBuffers.hidden } },
          { binding: 2, resource: { buffer: this.uniformBuffers.embeddingLookup } },
        ],
      }),
      finalNorm: this.device.createBindGroup({
        layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.hidden } },
          { binding: 1, resource: { buffer: this.modelBuffers.finalNorm! } },
          { binding: 2, resource: { buffer: this.workBuffers.normed } },
          { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
        ],
      }),
      lmHead: this.device.createBindGroup({
        layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.normed } },
          // Untied LM head if loaded (Qwen3 4B/8B), else tied embeddings.
          { binding: 1, resource: { buffer: this.modelBuffers.globals['lm_head'] ?? this.modelBuffers.tokenEmbed! } },
          { binding: 2, resource: { buffer: this.workBuffers.logits } },
          { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_V_H } },
        ],
      }),
      logitSoftcap: this.device.createBindGroup({
        layout: this.pipelines.logitSoftcap.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.logits } },
          { binding: 1, resource: { buffer: this.uniformBuffers.logitSoftcap } },
        ],
      }),
      argmax: this.device.createBindGroup({
        layout: this.pipelines.argmax.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.logits } },
          { binding: 1, resource: { buffer: this.workBuffers.argmaxResult } },
          { binding: 2, resource: { buffer: this.uniformBuffers.argmaxSize } },
        ],
      }),
      topk256: this.device.createBindGroup({
        layout: this.pipelines.topk256.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.logits } },
          { binding: 1, resource: { buffer: this.workBuffers.topk256Result } },
          { binding: 2, resource: { buffer: this.uniformBuffers.argmaxSize } },
        ],
      }),
      layers: [],
      plePmProjMatmul,
      pleStage1Fuse,
    };

    for (let i = 0; i < this.config.num_layers; i++) {
      const layer = this.modelBuffers.layers[i];
      // Consumer layers read K/V from their producer's cache rather than their own.
      // isKvProducerLayer(i) === true means kv_producer_for_layer[i] === i.
      const kvSourceIdx = this.config.kv_producer_for_layer[i];
      const kv = this.kvCaches[kvSourceIdx];
      const ownKv = this.kvCaches[i];
      const lb: LayerBindGroups = {
        attnNorm: this.device.createBindGroup({
          layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.hidden } },
            { binding: 1, resource: { buffer: layer.attn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.normed } },
            { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        linearQ: this.weightQuant !== 'f16'
          ? this.mmQuant(this.workBuffers.normed, layer, 'attn_q', this.workBuffers.q, this.uniformBuffers.linearQ8_Q_H[i])
          : this.device.createBindGroup({
          layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.attn_q } },
            { binding: 2, resource: { buffer: this.workBuffers.q } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_Q_H[i] } },
          ],
        }),
        linearK: this.weightQuant !== 'f16'
          ? this.mmQuant(this.workBuffers.normed, layer, 'attn_k', this.workBuffers.k, this.uniformBuffers.linearQ8_KV_H[i])
          : this.device.createBindGroup({
          layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.attn_k } },
            { binding: 2, resource: { buffer: this.workBuffers.k } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_KV_H[i] } },
          ],
        }),
        linearV: this.weightQuant !== 'f16'
          ? this.mmQuant(this.workBuffers.normed, layer, 'attn_v', this.workBuffers.v, this.uniformBuffers.linearQ8_KV_H[i])
          : this.device.createBindGroup({
          layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.attn_v } },
            { binding: 2, resource: { buffer: this.workBuffers.v } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_KV_H[i] } },
          ],
        }),
        ropeQ: this.device.createBindGroup({
          layout: this.pipelines.rope.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.q } },
            { binding: 1, resource: { buffer: this.uniformBuffers.ropeQ[i] } },
          ],
        }),
        ropeK: this.device.createBindGroup({
          layout: this.pipelines.rope.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.k } },
            { binding: 1, resource: { buffer: this.uniformBuffers.ropeK[i] } },
          ],
        }),
        qNorm: this.device.createBindGroup({
          layout: this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.q } },
            { binding: 1, resource: { buffer: layer.attn_q_norm ?? layer.attn_norm } },
            { binding: 2, resource: { buffer: this.uniformBuffers.perHeadRmsNormQ[i] } },
          ],
        }),
        kNorm: this.device.createBindGroup({
          layout: this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.k } },
            { binding: 1, resource: { buffer: layer.attn_k_norm ?? layer.attn_norm } },
            { binding: 2, resource: { buffer: this.uniformBuffers.perHeadRmsNormK[i] } },
          ],
        }),
        // Gemma 4 v_norm. Uses the no-weight variant because HF's v_norm has weight ≡ 1
        // (no learned param; attn_v_norm absent from the GGUF). Dispatched NKH times per
        // producer layer with HD = getHeadDim(layer) (256 sliding / 512 full).
        vNorm: this.device.createBindGroup({
          layout: this.pipelines.perHeadRmsNormNoWeight.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.v } },
            { binding: 1, resource: { buffer: this.uniformBuffers.perHeadRmsNormV[i] } },
          ],
        }),
        fusedNormRopeQ: this.device.createBindGroup({
          layout: this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.q } },
            { binding: 1, resource: { buffer: layer.attn_q_norm ?? layer.attn_norm } },
            { binding: 2, resource: { buffer: this.uniformBuffers.fusedNormRopeQ[i] } },
            // Qwen3 has no rope_freqs table (apply_divisor=0 ⇒ never read); bind a
            // present F16 buffer to satisfy the layout.
            { binding: 3, resource: { buffer: this.modelBuffers.globals.rope_freqs ?? layer.attn_q_norm ?? layer.attn_norm } },
          ],
        }),
        fusedNormRopeK: this.device.createBindGroup({
          layout: this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.k } },
            { binding: 1, resource: { buffer: layer.attn_k_norm ?? layer.attn_norm } },
            { binding: 2, resource: { buffer: this.uniformBuffers.fusedNormRopeK[i] } },
            { binding: 3, resource: { buffer: this.modelBuffers.globals.rope_freqs ?? layer.attn_k_norm ?? layer.attn_norm } },
          ],
        }),
        // kvStore writes to this layer's own cache (meaningless for consumer layers; they
        // skip this dispatch entirely in encodeTransformerPass).
        kvStore: this.device.createBindGroup({
          layout: this.pipelines.kvCacheStore.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.k } },
            { binding: 1, resource: { buffer: this.workBuffers.v } },
            { binding: 2, resource: { buffer: ownKv.k } },
            { binding: 3, resource: { buffer: ownKv.v } },
            { binding: 4, resource: { buffer: this.uniformBuffers.kvCacheStore[i] } },
          ],
        }),
        attnScore: this.device.createBindGroup({
          layout: this.pipelines.attnScore.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.q } },
            { binding: 1, resource: { buffer: kv.k } },
            { binding: 2, resource: { buffer: this.workBuffers.attnScores } },
            { binding: 3, resource: { buffer: this.uniformBuffers.attnScore[i] } },
          ],
        }),
        softmax: this.device.createBindGroup({
          layout: this.pipelines.softmax.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnScores } },
            { binding: 1, resource: { buffer: this.uniformBuffers.softmax } },
          ],
        }),
        attnOutput: this.device.createBindGroup({
          layout: this.pipelines.attnOutput.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnScores } },
            { binding: 1, resource: { buffer: kv.v } },
            { binding: 2, resource: { buffer: this.workBuffers.attnOut } },
            { binding: 3, resource: { buffer: this.uniformBuffers.attnOutput[i] } },
          ],
        }),
        linearAttnOut: this.weightQuant !== 'f16'
          ? this.mmQuant(this.workBuffers.attnOut, layer, 'attn_output', this.workBuffers.attnProj, this.uniformBuffers.linearQ8_H_Q[i])
          : this.device.createBindGroup({
          layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnOut } },
            { binding: 1, resource: { buffer: layer.attn_output } },
            { binding: 2, resource: { buffer: this.workBuffers.attnProj } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_H_Q[i] } },
          ],
        }),
        postAttnNorm: this.device.createBindGroup({
          layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnProj } },
            { binding: 1, resource: { buffer: layer.post_attention_norm ?? layer.attn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.postAttnNormed } },
            { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        residualAdd1: this.device.createBindGroup({
          layout: this.pipelines.add.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.hidden } },
            { binding: 1, resource: { buffer: this.workBuffers.postAttnNormed } },
            { binding: 2, resource: { buffer: this.workBuffers.residual } },
            { binding: 3, resource: { buffer: this.uniformBuffers.sizeH } },
          ],
        }),
        ffnNorm: this.device.createBindGroup({
          layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.residual } },
            { binding: 1, resource: { buffer: layer.ffn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.normed } },
            { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        // ffnGate/ffnUp bind-group layout comes from whichever matmul
        // pipeline will actually dispatch the kernel, selected from the
        // tuning profile at init. `layout: 'auto'` gives each pipeline a
        // unique layout object even when the struct is identical, so bind
        // groups are not interchangeable across pipelines — match exactly.
        ffnGate: this.weightQuant !== 'f16'
          ? this.mmQuant(this.workBuffers.normed, layer, 'ffn_gate', this.workBuffers.ffnGate, this.uniformBuffers.linearQ8_I_H[i])
          : this.device.createBindGroup({
          layout: (this.mr4ForFfn ? this.pipelines.matmulQuantMR4 : this.pipelines.matmulQuant).getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.ffn_gate } },
            { binding: 2, resource: { buffer: this.workBuffers.ffnGate } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_I_H[i] } },
          ],
        }),
        ffnUp: this.weightQuant !== 'f16'
          ? this.mmQuant(this.workBuffers.normed, layer, 'ffn_up', this.workBuffers.ffnUp, this.uniformBuffers.linearQ8_I_H[i])
          : this.device.createBindGroup({
          layout: (this.mr4ForFfn ? this.pipelines.matmulQuantMR4 : this.pipelines.matmulQuant).getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.ffn_up } },
            { binding: 2, resource: { buffer: this.workBuffers.ffnUp } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_I_H[i] } },
          ],
        }),
        // Activation bind group. Built from whichever pipeline the forward pass
        // will dispatch (auto-layouts aren't interchangeable across pipelines).
        geluMul: this.device.createBindGroup({
          layout: (this.config.ffn_activation === 'silu' ? this.pipelines.siluMul : this.pipelines.geluMul).getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.ffnGate } },
            { binding: 1, resource: { buffer: this.workBuffers.ffnUp } },
            { binding: 2, resource: { buffer: this.workBuffers.ffnMul } },
            { binding: 3, resource: { buffer: this.uniformBuffers.sizeI[i] } },
          ],
        }),
        ffnDown: this.config.arch === 'deepseek-ocr'
          ? this.mmWith(this.pipelines.matmulQ8Mr, this.workBuffers.ffnMul, layer, 'ffn_down', this.workBuffers.ffnDown, this.uniformBuffers.linearQ8_H_I[i])
          : this.weightQuant !== 'f16'
          ? this.mmQuant(this.workBuffers.ffnMul, layer, 'ffn_down', this.workBuffers.ffnDown, this.uniformBuffers.linearQ8_H_I[i])
          : this.device.createBindGroup({
          layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.ffnMul } },
            { binding: 1, resource: { buffer: layer.ffn_down } },
            { binding: 2, resource: { buffer: this.workBuffers.ffnDown } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_H_I[i] } },
          ],
        }),
        postFfnNorm: this.device.createBindGroup({
          layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.ffnDown } },
            { binding: 1, resource: { buffer: layer.post_ffw_norm ?? layer.ffn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.postFfnNormed } },
            { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        residualAdd2: this.device.createBindGroup({
          layout: this.pipelines.add.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.residual } },
            { binding: 1, resource: { buffer: this.workBuffers.postFfnNormed } },
            { binding: 2, resource: { buffer: this.workBuffers.hidden } },
            { binding: 3, resource: { buffer: this.uniformBuffers.sizeH } },
          ],
        }),
        fusedPostAttnNormAdd: this.device.createBindGroup({
          layout: this.pipelines.fusedNormAdd.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnProj } },
            { binding: 1, resource: { buffer: layer.post_attention_norm ?? layer.attn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.hidden } },
            { binding: 3, resource: { buffer: this.workBuffers.residual } },
            { binding: 4, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        fusedPostFfnNormAdd: this.device.createBindGroup({
          layout: this.pipelines.fusedNormAdd.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.ffnDown } },
            { binding: 1, resource: { buffer: layer.post_ffw_norm ?? layer.ffn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.residual } },
            { binding: 3, resource: { buffer: this.workBuffers.hidden } },
            { binding: 4, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        // PLE Stage 2 bind groups — five per layer.
        // 2a: gate = inp_gate @ hidden (post-FFN-residual stream).
        pleInpGateMatmul: this.device.createBindGroup({
          layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.hidden } },
            { binding: 1, resource: { buffer: layer.inp_gate ?? layer.attn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.pleGate } },
            { binding: 3, resource: { buffer: this.uniformBuffers.pleInpGateMM } },
          ],
        }),
        // 2b: gated = gelu(gate) * pleInputs[il].
        pleGeluMul: this.device.createBindGroup({
          layout: this.pipelines.pleGeluMul.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.pleGate } },
            { binding: 1, resource: { buffer: this.workBuffers.pleInputs } },
            { binding: 2, resource: { buffer: this.workBuffers.pleGated } },
            { binding: 3, resource: { buffer: this.uniformBuffers.pleGeluMulParams[i] } },
          ],
        }),
        // 2c: projOut = proj @ gated (per_layer_input_dim → hidden_size).
        plePostProjMatmul: this.device.createBindGroup({
          layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.pleGated } },
            { binding: 1, resource: { buffer: layer.proj ?? layer.attn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.pleProjOut } },
            { binding: 3, resource: { buffer: this.uniformBuffers.plePostProjMM } },
          ],
        }),
        // 2d: post_normed = rmsNorm(projOut) with layer.post_norm weight.
        plePostNorm: this.device.createBindGroup({
          layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.pleProjOut } },
            { binding: 1, resource: { buffer: layer.post_norm ?? layer.attn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.plePostNormed } },
            { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        // 2e: hidden = (hidden + plePostNormed) * layer_output_scale.
        pleSkipScaleAdd: this.device.createBindGroup({
          layout: this.pipelines.pleSkipScaleAdd.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.hidden } },
            { binding: 1, resource: { buffer: this.workBuffers.plePostNormed } },
            { binding: 2, resource: { buffer: layer.layer_output_scale ?? layer.attn_norm } },
            { binding: 3, resource: { buffer: this.uniformBuffers.pleSkipScaleAdd } },
          ],
        }),
        // Qwen3 pre-norm residual adds (replace Gemma's fused post-norm+add).
        // qwenAttnAdd: residual = hidden + attnProj.  qwenFfnAdd: hidden = residual + ffnDown.
        // Created for every arch (work-buffer-only; unused on the Gemma path).
        qwenAttnAdd: this.device.createBindGroup({
          layout: this.pipelines.add.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.hidden } },
            { binding: 1, resource: { buffer: this.workBuffers.attnProj } },
            { binding: 2, resource: { buffer: this.workBuffers.residual } },
            { binding: 3, resource: { buffer: this.uniformBuffers.sizeH } },
          ],
        }),
        qwenFfnAdd: this.device.createBindGroup({
          layout: this.pipelines.add.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.residual } },
            { binding: 1, resource: { buffer: this.workBuffers.ffnDown } },
            { binding: 2, resource: { buffer: this.workBuffers.hidden } },
            { binding: 3, resource: { buffer: this.uniformBuffers.sizeH } },
          ],
        }),
      };
      // MoE bind groups (deepseek-ocr routed layers). The shared-expert FFN
      // rides the standard ffnGate/ffnUp/geluMul/ffnDown groups above via
      // tensor aliasing; these cover the router + routed-expert pipeline.
      if (isMoELayer(i, this.config)) {
        const wb = this.workBuffers;
        const ub = this.uniformBuffers;
        lb.moeRouter = this.device.createBindGroup({
          layout: this.pipelines.matmulQuant.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: wb.normed } },
            { binding: 1, resource: { buffer: layer.ffn_gate_inp } },
            { binding: 2, resource: { buffer: wb.moeRouterLogits! } },
            { binding: 3, resource: { buffer: ub.moeRouterMM! } },
          ],
        });
        lb.moeTopk = this.device.createBindGroup({
          layout: this.pipelines.moeTopk.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: wb.moeRouterLogits! } },
            { binding: 1, resource: { buffer: wb.moeSel! } },
            { binding: 2, resource: { buffer: ub.moeTopkParams! } },
          ],
        });
        const expertBG = (key: string, out: GPUBuffer, params: GPUBuffer, pipe: GPUComputePipeline, input: GPUBuffer) =>
          this.device.createBindGroup({
            layout: pipe.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: input } },
              { binding: 1, resource: { buffer: layer[key] } },
              { binding: 2, resource: { buffer: layer[`${key}__s`] } },
              { binding: 3, resource: { buffer: out } },
              { binding: 4, resource: { buffer: params } },
              { binding: 5, resource: { buffer: wb.moeSel! } },
            ],
          });
        lb.moeGateExp = expertBG('ffn_gate_exps', wb.ffnGate, ub.moeGateUpMM!, this.pipelines.matmulQ4KExpert, wb.normed);
        lb.moeUpExp = expertBG('ffn_up_exps', wb.ffnUp, ub.moeGateUpMM!, this.pipelines.matmulQ4KExpert, wb.normed);
        lb.moeActMul = this.device.createBindGroup({
          layout: this.pipelines.siluMul.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: wb.ffnGate } },
            { binding: 1, resource: { buffer: wb.ffnUp } },
            { binding: 2, resource: { buffer: wb.ffnMul } },
            { binding: 3, resource: { buffer: ub.sizeMoeAct! } },
          ],
        });
        lb.moeDownExp = expertBG('ffn_down_exps', wb.moeSlots!, ub.moeDownMM!, this.pipelines.matmulQ8Expert, wb.ffnMul);
        lb.moeAccum = this.device.createBindGroup({
          layout: this.pipelines.moeAccum.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: wb.moeSlots! } },
            { binding: 1, resource: { buffer: wb.moeSel! } },
            { binding: 2, resource: { buffer: wb.ffnDown } },
            { binding: 3, resource: { buffer: ub.moeAccumParams! } },
          ],
        });
      }
      bgc.layers.push(lb);
    }
    this.bindGroupCache = bgc;
  }

  /** Encode transformer layers into a command encoder (shared between forward pass variants).
   *
   * `tokenId === null` is the pipelined-decode path (Option B): skip the CPU
   * writeBuffers for the embeddingLookup and pleStage1 uniforms because the
   * previous forward's sampleNextTokenSubmit landed the sampled tokenId
   * on-GPU via copyBufferToBuffer(argmaxResult → uniform). Letting us
   * encode+submit this forward without knowing the tokenId on CPU.
   */
  private encodeTransformerPass(
    encoder: GPUCommandEncoder,
    tokenId: number | null,
    position: number,
    stopAt?: CapturePoint,
  ): void {
    const C = this.config;
    const H = C.hidden_size;
    const NQH = C.num_q_heads;
    const NKH = C.num_kv_heads;
    const seqLen = position + 1;

    // Reset profile slot cursor at pipeline start. Guards against leaking
    // unresolved samples from a prior encoder (e.g. a capture diagnostic that
    // didn't call appendProfileResolve) into this pipeline's slot range.
    this.profileSlotCursor = 0;
    this.profilePassLabels = [];

    const cpuProf = this.cpuProfileActive && this.cpuProfileCapturing;
    const t_encodeStart = cpuProf ? performance.now() : 0;

    if (tokenId !== null) {
      this.wb(this.uniformBuffers.embeddingLookup, 4, new Uint32Array([tokenId]));
    }
    const posU32 = new Uint32Array([position]);
    const seqU32 = new Uint32Array([seqLen]);
    const tokenU32 = tokenId !== null ? new Uint32Array([tokenId]) : null;
    for (let il = 0; il < C.num_layers; il++) {
      if (C.qk_norm === false) {
        // Plain-rope path (deepseek-ocr): rope.wgsl Params.position at offset 8.
        this.wb(this.uniformBuffers.ropeQ[il], 8, posU32);
        this.wb(this.uniformBuffers.ropeK[il], 8, posU32);
      }
      this.wb(this.uniformBuffers.fusedNormRopeQ[il], 12, posU32);
      this.wb(this.uniformBuffers.fusedNormRopeK[il], 12, posU32);
      this.wb(this.uniformBuffers.kvCacheStore[il], 8, posU32);
      this.wb(this.uniformBuffers.attnScore[il], 12, seqU32);
      this.wb(this.uniformBuffers.attnOutput[il], 12, seqU32);
      if (tokenU32 !== null) {
        // pleStage1 uniform is { layer_idx, token_id, per_layer_dim, eps } — offset 4 = token_id.
        this.wb(this.uniformBuffers.pleStage1[il], 4, tokenU32);
      }
    }
    this.wb(this.uniformBuffers.softmax, 4, seqU32);

    const t_afterUniforms = cpuProf ? performance.now() : 0;

    let pass: GPUComputePassEncoder;

    pass = this.beginPass(encoder, 'embed');
    pass.setPipeline(this.pipelines.embeddingLookup);
    pass.setBindGroup(0, this.bindGroupCache.embeddingLookup);
    pass.dispatchWorkgroups(Math.ceil(H / 256));
    pass.end();

    if (stopAt?.kind === 'embed') return;

    // Per-layer embedding Stage 1 (runs once, before layer 0). Produces
    // workBuffers.pleInputs [num_layers × per_layer_input_dim]. Gemma 4 only —
    // Qwen3 has no PLE.
    if (C.arch === undefined || C.arch === 'gemma4') {
      const pleOut = C.num_layers * C.per_layer_input_dim;
      pass = this.beginPass(encoder, 'ple1.pmProj');
      pass.setPipeline(this.pipelines.matmulQuant);
      pass.setBindGroup(0, this.bindGroupCache.plePmProjMatmul!);
      this.dispatchMatmul(pass, pleOut);
      pass.end();

      if (stopAt?.kind === 'plePmProjected') return;

      pass = this.beginPass(encoder, 'ple1.stage1Fuse');
      pass.setPipeline(this.pipelines.pleStage1Fuse);
      for (let il = 0; il < C.num_layers; il++) {
        pass.setBindGroup(0, this.bindGroupCache.pleStage1Fuse[il]);
        pass.dispatchWorkgroups(1);
      }
      pass.end();

      if (stopAt?.kind === 'pleStage1') return;
    }

    for (let layerIdx = 0; layerIdx < C.num_layers; layerIdx++) {
      const lb = this.bindGroupCache.layers[layerIdx];
      const HD = getHeadDim(layerIdx, C);
      const I = getIntermediateSize(layerIdx, C);
      const Q = NQH * HD;
      const KV = NKH * HD;
      const isProducer = isKvProducerLayer(layerIdx, C);
      // Layer matmul weights are F16 (matmulQuant) or in-shader quant (Q8 / Q4_K).
      // Per-mode multi-row matmul: f16 → MR4 (when the profile enables it),
      // q8 → Q8 MR, q4k → Q4_K MR. All share their scalar counterpart's bind
      // layout, so the cached bind groups work unchanged.
      const mmPipe = this.weightQuant === 'q8' ? this.pipelines.matmulQ8Mr
        : this.weightQuant === 'q4k' ? this.pipelines.matmulQ4KMr
        : this.mr4ForFfn ? this.pipelines.matmulQuantMR4
        : this.pipelines.matmulQuant;
      pass = this.beginPass(encoder, 'attn.rmsNorm'); pass.setPipeline(this.pipelines.rmsNorm); pass.setBindGroup(0, lb.attnNorm); pass.dispatchWorkgroups(1); pass.end();
      pass = this.beginPass(encoder, isProducer ? 'attn.linearQKV.producer' : 'attn.linearQ.consumer');
      pass.setPipeline(mmPipe);
      pass.setBindGroup(0, lb.linearQ); this.dispatchMatmulRows(pass, Q);
      if (isProducer) {
        pass.setBindGroup(0, lb.linearK); this.dispatchMatmulRows(pass, KV);
        pass.setBindGroup(0, lb.linearV); this.dispatchMatmulRows(pass, KV);
      }
      pass.end();
      if (stopAt && 'layer' in stopAt && stopAt.layer === layerIdx) {
        if (stopAt.kind === 'preRopeQ' || stopAt.kind === 'preRopeK' || stopAt.kind === 'preRopeV') return;
      }
      // Gemma 4 v_norm: RMS-normalize V per KV-head (weight ≡ 1) before the cache store
      // so cached V (shared by consumer layers) is already normalized. Producer-only;
      // consumers read the normalized V straight from the shared cache. Must run before
      // `kvCacheStore` below.
      if (isProducer && C.v_norm !== false) {
        pass = this.beginPass(encoder, 'attn.vNorm');
        pass.setPipeline(this.pipelines.perHeadRmsNormNoWeight);
        pass.setBindGroup(0, lb.vNorm);
        pass.dispatchWorkgroups(NKH);
        pass.end();
      }
      if (C.qk_norm === false) {
        // deepseek-ocr: plain Llama RoPE on raw q/k — no per-head norm.
        // rope.wgsl covers num_heads*head_dim/2 rotation pairs per dispatch.
        pass = this.beginPass(encoder, isProducer ? 'attn.ropeQK' : 'attn.ropeQ');
        pass.setPipeline(this.pipelines.rope);
        pass.setBindGroup(0, lb.ropeQ); pass.dispatchWorkgroups(Math.ceil((NQH * HD / 2) / 256));
        if (isProducer) {
          pass.setBindGroup(0, lb.ropeK); pass.dispatchWorkgroups(Math.ceil((NKH * HD / 2) / 256));
        }
        pass.end();
      } else {
      pass = this.beginPass(encoder, isProducer ? 'attn.fusedNormRopeQK' : 'attn.fusedNormRopeQ');
      pass.setPipeline(this.pipelines.fusedPerHeadNormRope);
      pass.setBindGroup(0, lb.fusedNormRopeQ); pass.dispatchWorkgroups(NQH);
      if (isProducer) {
        pass.setBindGroup(0, lb.fusedNormRopeK); pass.dispatchWorkgroups(NKH);
      }
      pass.end();
      }
      if (stopAt && 'layer' in stopAt && stopAt.layer === layerIdx) {
        if (stopAt.kind === 'postRopeQ' || stopAt.kind === 'postRopeK') return;
      }
      if (isProducer) {
        pass = this.beginPass(encoder, 'attn.kvCacheStore'); pass.setPipeline(this.pipelines.kvCacheStore); pass.setBindGroup(0, lb.kvStore); pass.dispatchWorkgroups(Math.ceil(KV / 256)); pass.end();
      }
      pass = this.beginPass(encoder, 'attn.attnScore'); pass.setPipeline(this.pipelines.attnScore); pass.setBindGroup(0, lb.attnScore); pass.dispatchWorkgroups(Math.ceil((NQH * seqLen) / 256)); pass.end();
      pass = this.beginPass(encoder, 'attn.softmax'); pass.setPipeline(this.pipelines.softmax); pass.setBindGroup(0, lb.softmax); pass.dispatchWorkgroups(NQH); pass.end();
      pass = this.beginPass(encoder, 'attn.attnOutput'); pass.setPipeline(this.pipelines.attnOutput); pass.setBindGroup(0, lb.attnOutput); pass.dispatchWorkgroups(Math.ceil((NQH * HD) / 256)); pass.end();
      if (stopAt && stopAt.kind === 'attnOut' && stopAt.layer === layerIdx) return;
      pass = this.beginPass(encoder, 'attn.linearOut'); pass.setPipeline(mmPipe); pass.setBindGroup(0, lb.linearAttnOut); this.dispatchMatmulRows(pass, H); pass.end();
      // Post-attention residual. Gemma: norm(attnProj)+hidden (fused). Qwen3: plain hidden+attnProj.
      if (C.post_attn_norm === false) {
        pass = this.beginPass(encoder, 'attn.residualAdd'); pass.setPipeline(this.pipelines.add); pass.setBindGroup(0, lb.qwenAttnAdd); pass.dispatchWorkgroups(Math.ceil(H / 256)); pass.end();
      } else {
        pass = this.beginPass(encoder, 'attn.postNormAdd'); pass.setPipeline(this.pipelines.fusedNormAdd); pass.setBindGroup(0, lb.fusedPostAttnNormAdd); pass.dispatchWorkgroups(1); pass.end();
      }
      pass = this.beginPass(encoder, 'ffn.rmsNorm'); pass.setPipeline(this.pipelines.rmsNorm); pass.setBindGroup(0, lb.ffnNorm); pass.dispatchWorkgroups(1); pass.end();
      // Unified: mmPipe is the right kernel per mode (f16 MR4 / q8 MR / q4k MR /
      // scalar), and dispatchMatmulRows issues ceil(I/rows) workgroups.
      pass = this.beginPass(encoder, 'ffn.linearGateUp');
      pass.setPipeline(mmPipe);
      pass.setBindGroup(0, lb.ffnGate); this.dispatchMatmulRows(pass, I);
      pass.setBindGroup(0, lb.ffnUp); this.dispatchMatmulRows(pass, I);
      pass.end();
      // FFN activation: Gemma tanh-GELU, Qwen3 SwiGLU/SiLU. Same bind-group layout.
      const actPipeline = C.ffn_activation === 'silu' ? this.pipelines.siluMul : this.pipelines.geluMul;
      pass = this.beginPass(encoder, 'ffn.actMul'); pass.setPipeline(actPipeline); pass.setBindGroup(0, lb.geluMul); pass.dispatchWorkgroups(Math.ceil(I / 256)); pass.end();
      const mmPipeDown = C.arch === 'deepseek-ocr' ? this.pipelines.matmulQ8Mr : mmPipe;
      pass = this.beginPass(encoder, 'ffn.linearDown'); pass.setPipeline(mmPipeDown); pass.setBindGroup(0, lb.ffnDown); this.dispatchMatmulRows(pass, H); pass.end();
      // Routed experts (deepseek-ocr MoE layers). The dense-FFN block above
      // just computed the SHARED-expert FFN into ffnDown (aliased tensors);
      // here the router picks top-k experts, the batched expert kernels
      // (z-dimension = slot) run gate/up/act/down for all k in four
      // dispatches, and moeAccum lands ffnDown += sum w_k*E_k(normed) — the
      // full DeepSeek-V2 y = shared(x) + sum w_k E_k(x) before the residual.
      if (isMoELayer(layerIdx, C)) {
        const moe = C.moe!;
        const mI = moe.moe_intermediate_size;
        const K = moe.experts_per_tok;
        pass = this.beginPass(encoder, 'moe.router'); pass.setPipeline(this.pipelines.matmulQuant); pass.setBindGroup(0, lb.moeRouter!); this.dispatchMatmul(pass, moe.num_experts); pass.end();
        pass = this.beginPass(encoder, 'moe.topk'); pass.setPipeline(this.pipelines.moeTopk); pass.setBindGroup(0, lb.moeTopk!); pass.dispatchWorkgroups(1); pass.end();
        // Expert kernels are R=4 multi-row; z = top-k slot.
        pass = this.beginPass(encoder, 'moe.expGateUp');
        pass.setPipeline(this.pipelines.matmulQ4KExpert);
        pass.setBindGroup(0, lb.moeGateExp!); pass.dispatchWorkgroups(Math.ceil(mI / 4), 1, K);
        pass.setBindGroup(0, lb.moeUpExp!); pass.dispatchWorkgroups(Math.ceil(mI / 4), 1, K);
        pass.end();
        pass = this.beginPass(encoder, 'moe.actMul'); pass.setPipeline(this.pipelines.siluMul); pass.setBindGroup(0, lb.moeActMul!); pass.dispatchWorkgroups(Math.ceil((K * mI) / 256)); pass.end();
        pass = this.beginPass(encoder, 'moe.expDown'); pass.setPipeline(this.pipelines.matmulQ8Expert); pass.setBindGroup(0, lb.moeDownExp!); pass.dispatchWorkgroups(Math.ceil(H / 4), 1, K); pass.end();
        pass = this.beginPass(encoder, 'moe.accum'); pass.setPipeline(this.pipelines.moeAccum); pass.setBindGroup(0, lb.moeAccum!); pass.dispatchWorkgroups(Math.ceil(H / 256)); pass.end();
      }
      // Post-FFN residual. Gemma: norm(ffnDown)+residual (fused). Qwen3: plain residual+ffnDown.
      if (C.post_ffn_norm === false) {
        pass = this.beginPass(encoder, 'ffn.residualAdd'); pass.setPipeline(this.pipelines.add); pass.setBindGroup(0, lb.qwenFfnAdd); pass.dispatchWorkgroups(Math.ceil(H / 256)); pass.end();
      } else {
        pass = this.beginPass(encoder, 'ffn.postNormAdd'); pass.setPipeline(this.pipelines.fusedNormAdd); pass.setBindGroup(0, lb.fusedPostFfnNormAdd); pass.dispatchWorkgroups(1); pass.end();
      }

      // Gemma 4 PLE Stage 2 (per-layer) + layer_output_scale. Qwen3 has no PLE.
      if (C.arch === undefined || C.arch === 'gemma4') {
        const P = C.per_layer_input_dim;
        pass = this.beginPass(encoder, 'ple2.linearInpGate'); pass.setPipeline(this.pipelines.matmulQuant); pass.setBindGroup(0, lb.pleInpGateMatmul); this.dispatchMatmul(pass, P); pass.end();
        pass = this.beginPass(encoder, 'ple2.geluMul'); pass.setPipeline(this.pipelines.pleGeluMul); pass.setBindGroup(0, lb.pleGeluMul); pass.dispatchWorkgroups(Math.ceil(P / 256)); pass.end();
        pass = this.beginPass(encoder, 'ple2.linearPostProj'); pass.setPipeline(this.pipelines.matmulQuant); pass.setBindGroup(0, lb.plePostProjMatmul); this.dispatchMatmul(pass, H); pass.end();
        pass = this.beginPass(encoder, 'ple2.rmsNorm'); pass.setPipeline(this.pipelines.rmsNorm); pass.setBindGroup(0, lb.plePostNorm); pass.dispatchWorkgroups(1); pass.end();
        pass = this.beginPass(encoder, 'ple2.skipScaleAdd'); pass.setPipeline(this.pipelines.pleSkipScaleAdd); pass.setBindGroup(0, lb.pleSkipScaleAdd); pass.dispatchWorkgroups(Math.ceil(H / 256)); pass.end();
      }

      if (stopAt?.kind === 'afterLayer' && stopAt.layer === layerIdx) return;
    }

    pass = this.beginPass(encoder, 'final.rmsNorm');
    pass.setPipeline(this.pipelines.rmsNorm);
    pass.setBindGroup(0, this.bindGroupCache.finalNorm);
    pass.dispatchWorkgroups(1);
    pass.end();

    if (stopAt?.kind === 'logits') {
      // Run LM head + softcap so captureHidden can read back the
      // vocab-sized logits vector for cross-lab comparison against an
      // external reference backend.
      const V = this.config.vocab_size;
      pass = this.beginPass(encoder, 'lmHead');
      pass.setPipeline(this.pipelines.matmulQuant);
      pass.setBindGroup(0, this.bindGroupCache.lmHead);
      this.dispatchMatmul(pass, V);
      pass.end();
      if (this.config.final_logit_softcapping > 0) {
        pass = this.beginPass(encoder, 'logitSoftcap');
        pass.setPipeline(this.pipelines.logitSoftcap);
        pass.setBindGroup(0, this.bindGroupCache.logitSoftcap);
        pass.dispatchWorkgroups(Math.ceil(V / 256));
        pass.end();
      }
    }

    if (cpuProf) {
      const t_encodeDone = performance.now();
      this.recordCpuPhase('cpu.encode.uniforms', t_afterUniforms - t_encodeStart);
      this.recordCpuPhase('cpu.encode.transformerPasses', t_encodeDone - t_afterUniforms);
    }
  }

  /** Encode LM head + sampling into encoder, submit, and read back the selected token. */
  /**
   * Encode lmHead + softcap + sample passes, submit the encoder, and return
   * an unresolved readback promise. Sync up to submit — does NOT await.
   *
   * Greedy-fast branch additionally lands `argmaxResult` into the
   * `embeddingLookup` uniform (offset 4) and every `pleStage1[il]` uniform
   * (offset 4), so the NEXT forward can be encoded/submitted WITHOUT
   * knowing the sampled tokenId on CPU. That's what enables depth-2
   * pipelining in engine.generate — submit forward N+1 before awaiting
   * forward N's readback. GPU compute hides the submit→callback scheduling
   * slack that otherwise sits inside mapAsync.wait.
   *
   * Topk branch keeps the mapAsync-blocking pattern (JS-side sort +
   * repPenalty handling can't pipeline cleanly). Caller is expected to
   * await the returned readback immediately.
   */
  private sampleNextTokenSubmit(
    encoder: GPUCommandEncoder,
    temperature: number,
    topP: number,
    repPenalty: number,
    allTokens: number[],
  ): { readback: Promise<number>; profilePromise: Promise<void> | null; useGreedyFast: boolean } {
    if (this.deviceLost) throw new Error('WebGPU device lost');

    const cpuProf = this.cpuProfileActive && this.cpuProfileCapturing;
    const t_sampleStart = cpuProf ? performance.now() : 0;

    const V = this.config.vocab_size;
    const C = this.config;
    let pass: GPUComputePassEncoder;

    pass = this.beginPass(encoder, 'lmHead');
    pass.setPipeline(this.pipelines.matmulQuant);
    pass.setBindGroup(0, this.bindGroupCache.lmHead);
    this.dispatchMatmul(pass, V);
    pass.end();

    // Final-logit softcap: `tanh(logits / cap) * cap` (Gemma 4 E2B: cap=30).
    // Stabilizes greedy sampling vs raw logits. Skipped if the config's
    // `final_logit_softcapping` is 0.
    if (this.config.final_logit_softcapping > 0) {
      pass = this.beginPass(encoder, 'logitSoftcap');
      pass.setPipeline(this.pipelines.logitSoftcap);
      pass.setBindGroup(0, this.bindGroupCache.logitSoftcap);
      pass.dispatchWorkgroups(Math.ceil(V / 256));
      pass.end();
    }

    // Greedy-fast depth-2 path is gated by the tuning profile. With
    // `decodeDepth: 1` or `greedyFastPath: false` the engine stays on the
    // serial path — same correctness, ~2.7 ms/token more wall time.
    const pipelinedOK = this.tuning.pipeline.decodeDepth === 2 && this.tuning.pipeline.greedyFastPath;
    const useGreedyFast = pipelinedOK && (temperature === 0 && repPenalty <= 1.0);
    // Ping-pong readback slot for greedy-fast. Declared outside the branch so
    // the readback async closure below can reference it regardless of which
    // branch ran (topk branch leaves it null and uses topk256Readback).
    let greedyReadbackSlot: GPUBuffer | null = null;
    if (useGreedyFast) {
      pass = this.beginPass(encoder, 'sample.argmax');
      pass.setPipeline(this.pipelines.argmax);
      pass.setBindGroup(0, this.bindGroupCache.argmax);
      pass.dispatchWorkgroups(1);
      pass.end();
      // Option B fan-out: land the sampled tokenId directly into the
      // embedding-lookup and pleStage1 uniforms the NEXT forward will read.
      // These copies are microsecond-scale and let us skip the CPU-side
      // writeBuffer of tokenId in encodeTransformerPass — which is what
      // enables depth-2 pipelining (submitting the next forward without
      // awaiting this one's readback).
      encoder.copyBufferToBuffer(this.workBuffers.argmaxResult, 0, this.uniformBuffers.embeddingLookup, 4, 4);
      for (let il = 0; il < C.num_layers; il++) {
        encoder.copyBufferToBuffer(this.workBuffers.argmaxResult, 0, this.uniformBuffers.pleStage1[il], 4, 4);
      }
      // Pick the ping-pong readback slot BEFORE submit. The readback closure
      // captures this specific buffer so concurrent pipelined forwards map
      // distinct buffers (WebGPU forbids overlapping maps on one buffer).
      greedyReadbackSlot = this.workBuffers.argmaxReadbackPool[this.argmaxReadbackIdx];
      this.argmaxReadbackIdx = (this.argmaxReadbackIdx ^ 1) as 0 | 1;
      encoder.copyBufferToBuffer(this.workBuffers.argmaxResult, 0, greedyReadbackSlot, 0, 4);
    } else {
      pass = this.beginPass(encoder, 'sample.topk256');
      pass.setPipeline(this.pipelines.topk256);
      pass.setBindGroup(0, this.bindGroupCache.topk256);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(this.workBuffers.topk256Result, 0, this.workBuffers.topk256Readback, 0, 256 * 2 * 4);
    }

    const t_afterSampleEncode = cpuProf ? performance.now() : 0;
    const profileResolvePending = this.appendProfileResolve(encoder);
    const t_afterProfileResolve = cpuProf ? performance.now() : 0;
    const cmd = encoder.finish();
    const t_afterFinish = cpuProf ? performance.now() : 0;
    this.device.queue.submit([cmd]);
    const t_afterSubmit = cpuProf ? performance.now() : 0;
    // Collect GPU timestamps from the same submit the sample waits on. Parallel
    // with the argmax/topk mapAsync below — both complete when the submit does.
    const profilePromise = profileResolvePending ? this.collectProfileSamples() : null;

    if (cpuProf) {
      this.recordCpuPhase('cpu.encode.samplePasses', t_afterSampleEncode - t_sampleStart);
      this.recordCpuPhase('cpu.appendProfileResolve', t_afterProfileResolve - t_afterSampleEncode);
      this.recordCpuPhase('cpu.encoder.finish', t_afterFinish - t_afterProfileResolve);
      this.recordCpuPhase('cpu.queue.submit', t_afterSubmit - t_afterFinish);
    }

    const readback = (async (): Promise<number> => {
      // mapAsync.wait measured from this promise's FIRST await (the mapAsync
      // call). In serial mode this equals GPU-forward time; in pipelined
      // mode (depth-2) the prior await of the previous forward's readback
      // already consumed most of the GPU time, so this measurement drops —
      // that's the Option B win visible in cpuProfile output.
      const t_awaitStart = cpuProf ? performance.now() : 0;
      let result: number;
      if (useGreedyFast) {
        const slot = greedyReadbackSlot!;
        try {
          await slot.mapAsync(GPUMapMode.READ);
        } catch (e) {
          throw new Error(`GPU readback failed (device lost?): ${e}`);
        }
        const t_afterMapAsync = cpuProf ? performance.now() : 0;
        const resultArray = new Uint32Array(slot.getMappedRange());
        result = resultArray[0];
        slot.unmap();
        const t_afterUnmap = cpuProf ? performance.now() : 0;
        if (cpuProf) {
          this.recordCpuPhase('cpu.mapAsync.wait', t_afterMapAsync - t_awaitStart);
          this.recordCpuPhase('cpu.readback.process', t_afterUnmap - t_afterMapAsync);
        }
      } else {
        try {
          await this.workBuffers.topk256Readback.mapAsync(GPUMapMode.READ);
        } catch (e) {
          throw new Error(`GPU readback failed (device lost?): ${e}`);
        }
        const t_afterMapAsync = cpuProf ? performance.now() : 0;
        const topkData = new Float32Array(this.workBuffers.topk256Readback.getMappedRange().slice(0));
        this.workBuffers.topk256Readback.unmap();

        const candidates = new Array(256);
        const topkU32 = new Uint32Array(topkData.buffer.slice(0));
        for (let i = 0; i < 256; i++) {
          candidates[i] = { val: topkData[i * 2], id: topkU32[i * 2 + 1] };
        }

        if (repPenalty > 1.0 && allTokens.length > 0) {
          const seen = new Set(allTokens);
          for (let i = 0; i < 256; i++) {
            if (seen.has(candidates[i].id)) {
              if (candidates[i].val > 0) candidates[i].val /= repPenalty;
              else candidates[i].val *= repPenalty;
            }
          }
        }

        candidates.sort((a: { val: number }, b: { val: number }) => b.val - a.val);
        if (temperature === 0) {
          result = candidates[0].id;
        } else {
          const maxLogit = candidates[0].val;
          let sumExp = 0;
          const probs = new Float32Array(256);
          for (let i = 0; i < 256; i++) {
            probs[i] = Math.exp((candidates[i].val - maxLogit) / temperature);
            sumExp += probs[i];
          }

          let cumProb = 0, cutoff = 256;
          for (let i = 0; i < 256; i++) {
            cumProb += probs[i] / sumExp;
            if (cumProb >= topP) { cutoff = i + 1; break; }
          }
          let subsetSum = 0;
          for (let i = 0; i < cutoff; i++) subsetSum += probs[i];
          let r = Math.random() * subsetSum;
          let picked = candidates[cutoff - 1].id;
          for (let i = 0; i < cutoff; i++) {
            r -= probs[i];
            if (r <= 0) { picked = candidates[i].id; break; }
          }
          result = picked;
        }
        if (cpuProf) {
          this.recordCpuPhase('cpu.mapAsync.wait', t_afterMapAsync - t_awaitStart);
        }
      }
      return result;
    })();

    return { readback, profilePromise, useGreedyFast };
  }

  /** Thin async wrapper around sampleNextTokenSubmit — submits and awaits. */
  private async sampleNextToken(
    encoder: GPUCommandEncoder,
    temperature: number,
    topP: number,
    repPenalty: number,
    allTokens: number[],
  ): Promise<number> {
    const pending = this.sampleNextTokenSubmit(encoder, temperature, topP, repPenalty, allTokens);
    const token = await pending.readback;
    if (pending.profilePromise) await pending.profilePromise;
    return token;
  }

  private async forwardPassAndGetToken(
    tokenId: number,
    position: number,
    temperature: number = 0,
    topP: number = 0.9,
    repPenalty: number = 1.0,
    allTokens: number[] = [],
  ): Promise<number> {
    const cpuProf = this.cpuProfileActive && this.cpuProfileCapturing;
    const t_fwdStart = cpuProf ? performance.now() : 0;
    if (cpuProf) {
      this.cpuWbCountThisForward = 0;
      this.cpuPassCountThisForward = 0;
      // Gap since previous forward ended — time the generator loop, yield
      // resumption, and consumer-side work consumed. Only recorded from the
      // 2nd forward onward (cpuPrevForwardEnd_ms starts at 0, which we treat
      // as "first token, nothing to measure").
      if (this.cpuPrevForwardEnd_ms > 0) {
        this.recordCpuPhase('cpu.betweenForwards', t_fwdStart - this.cpuPrevForwardEnd_ms);
      }
    }
    const encoder = this.device.createCommandEncoder();
    this.encodeTransformerPass(encoder, tokenId, position);
    const result = await this.sampleNextToken(encoder, temperature, topP, repPenalty, allTokens);
    if (cpuProf) {
      const t_fwdEnd = performance.now();
      this.cpuForwardSamples_ms.push(t_fwdEnd - t_fwdStart);
      this.cpuForwardWriteBufferCounts.push(this.cpuWbCountThisForward);
      this.cpuForwardPassEncodeCounts.push(this.cpuPassCountThisForward);
      this.cpuPrevForwardEnd_ms = t_fwdEnd;
    }
    return result;
  }

  private forwardPassOnly(tokenId: number, position: number): void {
    const encoder = this.device.createCommandEncoder();
    this.encodeTransformerPass(encoder, tokenId, position);
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Non-awaiting variant for depth-2 pipelined decoding (Option B).
   * Encodes + submits one forward; returns the pending readback promise.
   *
   * Greedy-fast only. Pass tokenId=null to skip the CPU writeBuffers for
   * embeddingLookup / pleStage1 — relies on the prior forward's on-GPU
   * argmax→uniform copy. Pass a real tokenId only for the first pipelined
   * forward (where we already know the token from prefill's first-token
   * sample).
   */
  private forwardPassSubmitOnly(
    tokenId: number | null,
    position: number,
    allTokens: number[],
  ): { readback: Promise<number>; profilePromise: Promise<void> | null } {
    const encoder = this.device.createCommandEncoder();
    this.encodeTransformerPass(encoder, tokenId, position);
    const pending = this.sampleNextTokenSubmit(encoder, 0, 0.9, 1.0, allTokens);
    return { readback: pending.readback, profilePromise: pending.profilePromise };
  }

  private async prefillBatched(tokens: number[], startPos: number = 0): Promise<void> {
    // Profiling targets per-token decode forward passes. Prefill goes through
    // a different cost profile (longer sequences, different seqLen scaling) so
    // we pause capture here even if the master switch is on. The caller
    // (lab.profile) re-enables capture after await returns.
    const wasCapturing = this.profileCapturing;
    const wasCpuCapturing = this.cpuProfileCapturing;
    this.profileCapturing = false;
    this.cpuProfileCapturing = false;
    for (let i = 0; i < tokens.length; i++) {
      this.forwardPassOnly(tokens[i], startPos + i);
    }
    await this.device.queue.onSubmittedWorkDone();
    this.profileCapturing = wasCapturing;
    this.cpuProfileCapturing = wasCpuCapturing;
  }

  private resetKVCaches(): void {
    const HD = this.config.head_dim;
    const NKH = this.config.num_kv_heads;
    const MAX_SEQ = this.config.context_length;
    const zeros = new Float32Array(MAX_SEQ * NKH * HD);
    for (let i = 0; i < this.config.num_layers; i++) {
      this.wb(this.kvCaches[i].k, 0, zeros);
      this.wb(this.kvCaches[i].v, 0, zeros);
    }
  }

  private async getFirstTokenAfterPrefill(
    temperature: number,
    topP: number,
    repPenalty: number,
    allTokens: number[],
  ): Promise<number> {
    const encoder = this.device.createCommandEncoder();
    return this.sampleNextToken(encoder, temperature, topP, repPenalty, allTokens);
  }

  // ─── Public API ────────────────────────────────────────────────

  addUserMessage(text: string): void {
    this.conversationHistory.push({ role: 'user', text });
  }

  async *generate(options: GenerateOptions = {}): AsyncGenerator<string, void, undefined> {
    if (this.deviceLost) throw new Error('WebGPU device lost — call dispose() and recreate the engine');

    const temperature = options.temperature ?? 0.7;
    const topP = options.topP ?? 0.9;
    const repPenalty = options.repPenalty ?? 1.2;
    const maxTokens = options.maxTokens ?? 32768;
    const toolsJson = options.toolsJson ?? '[]';
    const signal = options.signal;

    // ── Generator-internal timing (see `lastGenerateStats`) ──────────
    // `workStart` tracks the timestamp at which the generator resumed
    // (post-yield) or began a new work segment. Before each yield we
    // accumulate `performance.now() - workStart` into `coreDecodeMs`.
    // The time between yield and resume — which the caller's `for await`
    // body consumes — is NOT counted. That gives us a pure-engine tps
    // number, directly comparable to `runBench`.
    const genStart = performance.now();
    let coreDecodeMs = 0;
    let workStart = genStart;
    let tokensYielded = 0;
    let lastYieldAt = genStart;
    let firstTokenMs = 0;
    let aborted = false;

    let newTokens: number[];
    if (this.kvPosition === 0) {
      const fullPrompt = buildChatPrompt(this.conversationHistory, toolsJson, this._systemPrompt);
      newTokens = this.tokenizer.encode(fullPrompt);
    } else {
      const lastUser = this.conversationHistory[this.conversationHistory.length - 1];
      const suffix = `<end_of_turn>\n<start_of_turn>user\n${lastUser.text}<end_of_turn>\n<start_of_turn>model\n`;
      newTokens = this.tokenizer.encode(suffix).slice(1); // Remove BOS
    }

    // Check context overflow — reset if needed
    if (this.kvPosition + newTokens.length >= this.config.context_length - 10) {
      const lastUser = this.conversationHistory[this.conversationHistory.length - 1];
      this.conversationHistory = [{ role: 'user', text: lastUser.text }];
      const freshPrompt = buildChatPrompt(this.conversationHistory, toolsJson, this._systemPrompt);
      newTokens = this.tokenizer.encode(freshPrompt);
      this.resetKVCaches();
      this.kvPosition = 0;
    }

    // Prefill
    await this.prefillBatched(newTokens, this.kvPosition);
    const allTokens = [...newTokens];
    this.kvPosition += newTokens.length;

    // Get first token (LM head on prefill output)
    let nextToken = await this.getFirstTokenAfterPrefill(temperature, topP, repPenalty, allTokens);
    allTokens.push(nextToken);

    const endFuncCall = this.tokenizer.funcTokens['<end_function_call>'];
    const generatedTokens: number[] = [nextToken];

    // First token. `firstTokenMs` captures prefill + first-token wall time.
    // After this yield, `coreDecodeMs` starts accumulating for subsequent
    // tokens only (matches `(tokens - 1) / decodeMs` in runBench).
    firstTokenMs = performance.now() - genStart;
    tokensYielded = 1;
    lastYieldAt = performance.now();
    yield this.tokenizer.decodeToken(nextToken);
    workStart = performance.now();

    // Greedy-fast depth-2 path is gated by the tuning profile. With
    // `decodeDepth: 1` or `greedyFastPath: false` the engine stays on the
    // serial path — same correctness, ~2.7 ms/token more wall time.
    const pipelinedOK = this.tuning.pipeline.decodeDepth === 2 && this.tuning.pipeline.greedyFastPath;
    const useGreedyFast = pipelinedOK && (temperature === 0 && repPenalty <= 1.0);
    const isHardStop = (t: number) => (t === 1 || t === 0 || t === 50 || t === 106);
    let committedKVWrites = 0;

    if (useGreedyFast) {
      // Depth-2 pipelined decode (Option B). Submit forward N+1 before
      // awaiting forward N's readback, so GPU compute hides the
      // submit→callback scheduling slack that sits inside mapAsync.wait in
      // the serial path (~2.7 ms/token on this 5090 setup).
      //
      // `pending` holds the readback promise for the most-recently submitted
      // forward whose token has not yet been committed. We maintain one
      // speculative forward in flight at all times until maxTokens or stop.
      //
      // KV bookkeeping: each submit writes KV at `kvPosition + submittedKVWrites`
      // (pre-increment). The committed count advances only after we accept
      // the token (not stop-discarded). A speculative forward submitted
      // before we saw a stop doesn't advance committedKVWrites — its KV
      // slot will be overwritten by the next generate call's prefill
      // (the conversation continues from kvPosition + committedKVWrites).
      let pending: { readback: Promise<number>; profilePromise: Promise<void> | null } | null = null;
      let submittedKVWrites = 0;

      const firstIsStop = isHardStop(nextToken) || (endFuncCall && nextToken === endFuncCall);
      if (!firstIsStop && maxTokens > 1 && !signal?.aborted) {
        const initPos = this.kvPosition + submittedKVWrites;
        if (initPos < this.config.context_length - 1) {
          // First pipelined forward — we know `nextToken` (= firstToken),
          // CPU-write it. Subsequent submits use tokenId=null and rely on
          // the preceding forward's on-GPU argmax→uniform copy.
          pending = this.forwardPassSubmitOnly(nextToken, initPos, allTokens);
          submittedKVWrites++;
        }
      }

      for (let step = 1; step < maxTokens && pending !== null; step++) {
        if (signal?.aborted) {
          await pending.readback;
          if (pending.profilePromise) await pending.profilePromise;
          pending = null;
          break;
        }

        // Speculatively submit the NEXT forward BEFORE awaiting current.
        // This is the actual depth-2 pipeline move.
        let nextPending: { readback: Promise<number>; profilePromise: Promise<void> | null } | null = null;
        const nextPos = this.kvPosition + submittedKVWrites;
        const canSubmitNext = (step + 1 < maxTokens) && (nextPos < this.config.context_length - 1);
        if (canSubmitNext) {
          nextPending = this.forwardPassSubmitOnly(null, nextPos, allTokens);
          submittedKVWrites++;
        }

        nextToken = await pending.readback;
        if (pending.profilePromise) await pending.profilePromise;
        pending = nextPending;

        if (isHardStop(nextToken)) {
          committedKVWrites++;
          if (pending) {
            await pending.readback;
            if (pending.profilePromise) await pending.profilePromise;
            pending = null;
          }
          break;
        }
        if (endFuncCall && nextToken === endFuncCall) {
          committedKVWrites++;
          allTokens.push(nextToken);
          generatedTokens.push(nextToken);
          if (pending) {
            await pending.readback;
            if (pending.profilePromise) await pending.profilePromise;
            pending = null;
          }
          break;
        }

        committedKVWrites++;
        allTokens.push(nextToken);
        generatedTokens.push(nextToken);

        coreDecodeMs += performance.now() - workStart;
        tokensYielded++;
        lastYieldAt = performance.now();
        yield this.tokenizer.decodeToken(nextToken);
        workStart = performance.now();
      }
    } else {
      // Serial loop (topk / sampled / repPenalty path). The JS-side sort +
      // repPenalty update can't pipeline cleanly, so stay serial.
      for (let step = 1; step < maxTokens; step++) {
        if (isHardStop(nextToken)) break;
        if (endFuncCall && nextToken === endFuncCall) break;
        if (signal?.aborted) break;

        const pos = this.kvPosition + committedKVWrites;
        if (pos >= this.config.context_length - 1) break;

        nextToken = await this.forwardPassAndGetToken(nextToken, pos, temperature, topP, repPenalty, allTokens);
        committedKVWrites++;

        if (isHardStop(nextToken)) break;
        if (endFuncCall && nextToken === endFuncCall) {
          allTokens.push(nextToken);
          generatedTokens.push(nextToken);
          break;
        }

        allTokens.push(nextToken);
        generatedTokens.push(nextToken);

        coreDecodeMs += performance.now() - workStart;
        tokensYielded++;
        lastYieldAt = performance.now();
        yield this.tokenizer.decodeToken(nextToken);
        workStart = performance.now();
      }
    }

    aborted = signal?.aborted === true;

    // Save model response and update KV position
    const rawOutput = this.tokenizer.decodeTokens(generatedTokens);
    this.conversationHistory.push({ role: 'model', text: rawOutput });
    this.kvPosition += committedKVWrites;

    // Finalize timing stats. `lastYieldAt` is the timestamp of the most
    // recent yield; `coreDecodeMs` is the summed in-generator work time
    // between the first and last yield (excluding caller body time).
    this._lastGenerateStats = {
      tokens: tokensYielded,
      firstTokenMs,
      coreDecodeMs,
      totalWallMs: lastYieldAt - genStart,
      aborted,
    };
  }

  resetConversation(): void {
    this.conversationHistory = [];
    this.kvPosition = 0;
    this.resetKVCaches();
  }

  /** Currently-applied system instruction, or null if none set. */
  get systemPrompt(): string | null {
    return this._systemPrompt;
  }

  /** Timing stats from the most recent `generate()` call. `null` until the
   *  first `generate()` has yielded at least one token. */
  get lastGenerateStats(): import('./types.js').LastGenerateStats | null {
    return this._lastGenerateStats;
  }

  /**
   * Set the system instruction emitted as a `developer`-role preamble.
   * Pass `null` or `''` to clear. Does NOT reset conversation state on its
   * own — callers should `resetConversation()` before the next `generate()`
   * for the new preamble to take effect (the current KV cache already has
   * the old preamble baked in).
   */
  setSystemPrompt(text: string | null): void {
    const trimmed = text?.trim() ?? '';
    this._systemPrompt = trimmed === '' ? null : trimmed;
  }

  // ─── Diagnostic: hidden-state capture ─────────────────────────────

  async captureHidden(
    tokenId: number,
    position: number,
    stopAt: CapturePoint,
  ): Promise<Float32Array> {
    if (this.deviceLost) throw new Error('WebGPU device lost');

    const encoder = this.device.createCommandEncoder();
    this.encodeTransformerPass(encoder, tokenId, position, stopAt);

    // Pick source buffer + byte count based on where we stopped. Most captures
    // use hiddenReadback (sized for the largest non-logits intermediate); the
    // V=262144-element logits path uses the dedicated logitsReadback instead.
    let src: GPUBuffer;
    let elemCount: number;
    let dst: GPUBuffer = this.workBuffers.hiddenReadback;
    switch (stopAt.kind) {
      case 'embed':
      case 'afterLayer':
        src = this.workBuffers.hidden;
        elemCount = this.config.hidden_size;
        break;
      case 'plePmProjected':
        src = this.workBuffers.plePmProjected;
        elemCount = this.config.num_layers * this.config.per_layer_input_dim;
        break;
      case 'pleStage1':
        src = this.workBuffers.pleInputs;
        elemCount = this.config.num_layers * this.config.per_layer_input_dim;
        break;
      case 'final':
        src = this.workBuffers.normed;
        elemCount = this.config.hidden_size;
        break;
      case 'logits':
        src = this.workBuffers.logits;
        elemCount = this.config.vocab_size;
        dst = this.workBuffers.logitsReadback;
        break;
      case 'preRopeQ':
      case 'postRopeQ':
      case 'attnOut': {
        const HD = getHeadDim(stopAt.layer, this.config);
        src = stopAt.kind === 'attnOut' ? this.workBuffers.attnOut : this.workBuffers.q;
        elemCount = this.config.num_q_heads * HD;
        break;
      }
      case 'preRopeK':
      case 'postRopeK': {
        const HD = getHeadDim(stopAt.layer, this.config);
        src = this.workBuffers.k;
        elemCount = this.config.num_kv_heads * HD;
        break;
      }
      case 'preRopeV': {
        const HD = getHeadDim(stopAt.layer, this.config);
        src = this.workBuffers.v;
        elemCount = this.config.num_kv_heads * HD;
        break;
      }
    }
    const byteSize = elemCount * 4;

    encoder.copyBufferToBuffer(src, 0, dst, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);

    await dst.mapAsync(GPUMapMode.READ, 0, byteSize);
    const mapped = dst.getMappedRange(0, byteSize);
    const result = new Float32Array(mapped.slice(0));
    dst.unmap();
    return result;
  }

  encodePromptTokens(text: string, applyChatTemplate: boolean): number[] {
    if (applyChatTemplate) {
      const fullPrompt = buildChatPrompt([{ role: 'user', text }], '[]');
      return this.tokenizer.encode(fullPrompt);
    }
    return this.tokenizer.encode(text);
  }

  async prefillForCapture(tokens: number[], startPos: number): Promise<void> {
    if (tokens.length === 0) return;
    await this.prefillBatched(tokens, startPos);
    this.kvPosition = startPos + tokens.length;
  }

  resetKVForCapture(): void {
    this.resetKVCaches();
    this.kvPosition = 0;
  }

  async captureEmbedSliceAll(tokenId: number): Promise<Float32Array[]> {
    if (this.deviceLost) throw new Error('WebGPU device lost');
    const D = this.config.per_layer_input_dim;
    if (D === 0) throw new Error('captureEmbedSliceAll: no PLE on this model');
    const N = this.config.num_layers;
    const bytesPerSlice = D * 2; // F16
    const totalBytes = N * bytesPerSlice;
    if (totalBytes > this.workBuffers.hiddenReadback.size) {
      throw new Error(`captureEmbedSliceAll: readback buffer too small (${totalBytes} > ${this.workBuffers.hiddenReadback.size})`);
    }

    const encoder = this.device.createCommandEncoder();
    const srcOffset = tokenId * bytesPerSlice;
    for (let L = 0; L < N; L++) {
      encoder.copyBufferToBuffer(
        this.modelBuffers.perLayerEmbeddings[L],
        srcOffset,
        this.workBuffers.hiddenReadback,
        L * bytesPerSlice,
        bytesPerSlice,
      );
    }
    this.device.queue.submit([encoder.finish()]);

    await this.workBuffers.hiddenReadback.mapAsync(GPUMapMode.READ, 0, totalBytes);
    const mapped = this.workBuffers.hiddenReadback.getMappedRange(0, totalBytes);
    // Copy out first — getMappedRange gives a view backed by the mapped region.
    const copy = mapped.slice(0);
    this.workBuffers.hiddenReadback.unmap();
    const f16 = new F16A(copy);

    const out: Float32Array[] = [];
    for (let L = 0; L < N; L++) {
      const slice = new Float32Array(D);
      for (let i = 0; i < D; i++) slice[i] = f16[L * D + i];
      out.push(slice);
    }
    return out;
  }

  // ─── Diagnostic: weight-layout probes ─────────────────────────────
  //
  // Shape question the probes answer: the `per_layer_model_projection`
  // tensor is logically 3D with JAX einsum axes
  // `d=hidden, n=layers, p=per_layer_dim` (`'...td,dnp->...tnp'`). GGUF
  // stores a 2D [hidden, num_layers * per_layer_input_dim] projection;
  // the uncertainty is the flatten order of the (n, p) axis
  // (`n*per_layer_dim + p` vs `p*num_layers + n`) and any transpose the
  // converter may have applied. `readGlobalTensor` + `fetchRawTensorSlice`
  // let a diagnostic tool compare the two side by side at a known index
  // and lock the layout down.

  getGgufTensorInfo(name: string): { dims: number[]; type: number; offset: number; byteSize: number } | null {
    if (!this.ggufTensors) return null;
    const t = this.ggufTensors.find(t => t.name === name + '.weight' || t.name === name);
    if (!t) return null;
    return {
      dims: t.dims.map(d => Number(d)),
      type: t.type,
      offset: Number(t.offset),
      byteSize: tensorByteSize(t),
    };
  }

  async readGlobalTensor(name: string, elemOffset: number, elemCount: number): Promise<Float32Array> {
    const buf = this.modelBuffers.globals[name];
    if (!buf) throw new Error(`readGlobalTensor: no global tensor '${name}' uploaded`);
    return this.readF16Buffer(buf, elemOffset, elemCount, `global:${name}`);
  }

  /** Diagnostic: read an F16 slice from a per-layer weight buffer. */
  async readLayerTensor(layer: number, name: string, elemOffset: number, elemCount: number): Promise<Float32Array> {
    const layers = this.modelBuffers.layers;
    if (layer < 0 || layer >= layers.length) {
      throw new Error(`readLayerTensor: layer ${layer} out of range [0, ${layers.length})`);
    }
    const buf = layers[layer][name];
    if (!buf) throw new Error(`readLayerTensor: no per-layer tensor '${name}' at layer ${layer}`);
    return this.readF16Buffer(buf, elemOffset, elemCount, `layer${layer}:${name}`);
  }

  /** Diagnostic: read an F16 slice from the tied embedding / LM head buffer. */
  async readEmbeddingTensor(elemOffset: number, elemCount: number): Promise<Float32Array> {
    const buf = this.modelBuffers.tokenEmbed;
    if (!buf) throw new Error('readEmbeddingTensor: embedding buffer not uploaded');
    return this.readF16Buffer(buf, elemOffset, elemCount, 'tokenEmbed');
  }

  /** Shared F16 readback path — handles 4-byte alignment for copyBufferToBuffer. */
  private async readF16Buffer(buf: GPUBuffer, elemOffset: number, elemCount: number, label: string): Promise<Float32Array> {
    if (this.deviceLost) throw new Error('WebGPU device lost');
    const byteOffset = elemOffset * 2;
    const byteCount = elemCount * 2;
    if (byteOffset + byteCount > buf.size) {
      throw new Error(`${label}: slice [${elemOffset}..${elemOffset + elemCount}) out of range (${buf.size / 2} elems)`);
    }
    const alignedSrcOffset = byteOffset & ~3;
    const headDrop = (byteOffset - alignedSrcOffset) / 2;
    const alignedByteCount = ((byteCount + (byteOffset - alignedSrcOffset) + 3) & ~3);

    const readback = this.device.createBuffer({
      size: alignedByteCount,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buf, alignedSrcOffset, readback, 0, alignedByteCount);
    this.device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();

    const f16 = new F16A(mapped, 0, headDrop + elemCount);
    const out = new Float32Array(elemCount);
    for (let i = 0; i < elemCount; i++) out[i] = f16[headDrop + i];
    return out;
  }

  async fetchRawTensorSlice(name: string, elemOffset: number, elemCount: number): Promise<Float32Array> {
    if (!this.ggufTensors || this.ggufDataOffset === undefined) {
      throw new Error('fetchRawTensorSlice: GGUF metadata not persisted');
    }
    if (!this.modelFile) {
      throw new Error('fetchRawTensorSlice: Range refetch unavailable (loaded from buffer, not URL)');
    }
    const t = this.ggufTensors.find(tt => tt.name === name + '.weight' || tt.name === name);
    if (!t) throw new Error(`fetchRawTensorSlice: no tensor named ${name}`);
    const numElems = Number(t.dims.reduce((a, b) => a * b, 1n));
    if (elemOffset < 0 || elemOffset + elemCount > numElems) {
      throw new Error(`fetchRawTensorSlice: [${elemOffset}..${elemOffset + elemCount}) exceeds tensor length ${numElems}`);
    }

    // Unquantized types: direct element-sliced fetch.
    if (t.type === 0 || t.type === 1 || t.type === 30) {
      const bytesPerElem = t.type === 0 ? 4 : 2;
      const absStart = this.ggufDataOffset + Number(t.offset) + elemOffset * bytesPerElem;
      const sliceBytes = elemCount * bytesPerElem;
      const resp = await fetch(this.modelFile, { headers: { Range: `bytes=${absStart}-${absStart + sliceBytes - 1}` } });
      if (!resp.ok && resp.status !== 206) {
        throw new Error(`fetchRawTensorSlice: HTTP ${resp.status} for range ${absStart}+${sliceBytes}`);
      }
      const bytes = new Uint8Array(await resp.arrayBuffer());
      if (t.type === 30) {
        const out = new Float32Array(elemCount);
        const ab = new ArrayBuffer(4);
        const u32 = new Uint32Array(ab);
        const f32 = new Float32Array(ab);
        const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        for (let i = 0; i < elemCount; i++) {
          u32[0] = dv.getUint16(i * 2, true) << 16;
          out[i] = f32[0];
        }
        return out;
      }
      if (t.type === 1) {
        const f16 = new F16A(bytes.buffer, bytes.byteOffset, elemCount);
        const out = new Float32Array(elemCount);
        for (let i = 0; i < elemCount; i++) out[i] = f16[i];
        return out;
      }
      return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + sliceBytes));
    }

    // Block-quantized types: fetch super-blocks covering the requested range,
    // dequant using a throwaway GGUFParser over the fetched bytes, then slice.
    // blockElems × blockBytes per type (llama.cpp GGML_QUANT_SIZES).
    const BLOCK: Record<number, { elems: number; bytes: number; dequant: (p: GGUFParser, off: number, n: number) => Float32Array }> = {
      6:  { elems: 32,  bytes: 22,  dequant: (p, o, n) => p.dequantizeQ5_0(o, n) },
      8:  { elems: 32,  bytes: 34,  dequant: (p, o, n) => p.dequantizeQ8_0(o, n) },
      12: { elems: 256, bytes: 144, dequant: (p, o, n) => p.dequantizeQ4_K(o, n) },
      13: { elems: 256, bytes: 176, dequant: (p, o, n) => p.dequantizeQ5_K(o, n) },
      14: { elems: 256, bytes: 210, dequant: (p, o, n) => p.dequantizeQ6_K(o, n) },
    };
    const spec = BLOCK[t.type];
    if (!spec) throw new Error(`fetchRawTensorSlice: unsupported source type ${t.type}`);

    const startBlock = Math.floor(elemOffset / spec.elems);
    const endBlock = Math.ceil((elemOffset + elemCount) / spec.elems);
    const numBlocks = endBlock - startBlock;
    const absStart = this.ggufDataOffset + Number(t.offset) + startBlock * spec.bytes;
    const fetchBytes = numBlocks * spec.bytes;
    const resp = await fetch(this.modelFile, { headers: { Range: `bytes=${absStart}-${absStart + fetchBytes - 1}` } });
    if (!resp.ok && resp.status !== 206) {
      throw new Error(`fetchRawTensorSlice: HTTP ${resp.status} for quant-range ${absStart}+${fetchBytes}`);
    }
    const bytes = new Uint8Array(await resp.arrayBuffer());
    const tempParser = new GGUFParser(bytes);
    const fullBlockElems = numBlocks * spec.elems;
    const decoded = spec.dequant(tempParser, 0, fullBlockElems);
    const sliceStart = elemOffset - startBlock * spec.elems;
    return decoded.subarray(sliceStart, sliceStart + elemCount).slice();
  }

  // ─── Profiling ─────────────────────────────────────────────────

  getProfileCapability(): ProfileCapability {
    return { ...this.profileCapability };
  }

  enableProfile(): void {
    if (!this.profileCapability.timestampQuerySupported) {
      this.profileActive = false;
      this.profileCapturing = false;
      return;
    }
    if (!this.profileQuerySet) {
      const slots = this.profileCapability.querySlots;
      this.profileQuerySet = this.device.createQuerySet({ type: 'timestamp', count: slots });
      this.profileResolveBuf = this.device.createBuffer({
        size: slots * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.profileStagingBuf = this.device.createBuffer({
        size: slots * 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
    }
    this.profileActive = true;
    this.profileCapturing = true;
  }

  disableProfile(): void {
    this.profileActive = false;
    this.profileCapturing = false;
  }

  resetProfileSamples(): void {
    this.profileSamples.clear();
    this.profileForwardTotals_ns.length = 0;
    this.profileOverflow = false;
  }

  enableCpuProfile(): void {
    this.cpuProfileActive = true;
    this.cpuProfileCapturing = true;
  }

  disableCpuProfile(): void {
    this.cpuProfileActive = false;
    this.cpuProfileCapturing = false;
  }

  resetCpuProfileSamples(): void {
    this.cpuPhaseSamples.clear();
    this.cpuForwardSamples_ms.length = 0;
    this.cpuForwardWriteBufferCounts.length = 0;
    this.cpuForwardPassEncodeCounts.length = 0;
    this.cpuPrevForwardEnd_ms = 0;
  }

  getCpuProfileReport(): CpuProfileReport {
    const perPhase: CpuProfileReport['perPhase'] = [];
    for (const [phase, samples] of this.cpuPhaseSamples) {
      if (samples.length === 0) continue;
      const sorted = samples.slice().sort((a, b) => a - b);
      const p50 = sorted[Math.floor(sorted.length * 0.5)];
      const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
      let sum = 0;
      for (const s of samples) sum += s;
      perPhase.push({
        phase,
        samples_n: samples.length,
        p50_ms: p50,
        p95_ms: p95,
        mean_ms: sum / samples.length,
        total_ms: sum,
      });
    }
    perPhase.sort((a, b) => b.p50_ms - a.p50_ms);
    const median = (arr: number[]): number => {
      if (arr.length === 0) return 0;
      const s = arr.slice().sort((a, b) => a - b);
      return s[Math.floor(s.length * 0.5)];
    };
    return {
      perPhase,
      forwardMedian_ms: median(this.cpuForwardSamples_ms),
      forwardSamples_n: this.cpuForwardSamples_ms.length,
      writeBufferCountMedian: median(this.cpuForwardWriteBufferCounts),
      passEncodeCountMedian: median(this.cpuForwardPassEncodeCounts),
    };
  }

  getProfileReport(): ProfileReport {
    const perLabel: ProfileReport['perLabel'] = [];
    for (const [label, samples] of this.profileSamples) {
      if (samples.length === 0) continue;
      const sorted = samples.slice().sort((a, b) => a - b);
      const p50 = sorted[Math.floor(sorted.length * 0.5)];
      const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
      let sum = 0;
      for (const s of samples) sum += s;
      perLabel.push({
        label,
        samples_n: samples.length,
        p50_ns: p50,
        p95_ns: p95,
        mean_ns: sum / samples.length,
        total_ns: sum,
      });
    }
    perLabel.sort((a, b) => b.p50_ns - a.p50_ns);
    const fwdSorted = this.profileForwardTotals_ns.slice().sort((a, b) => a - b);
    const forwardMedian_ns = fwdSorted.length > 0 ? fwdSorted[Math.floor(fwdSorted.length * 0.5)] : 0;
    return {
      perLabel,
      forwardMedian_ns,
      forwardSamples_n: this.profileForwardTotals_ns.length,
      overflow: this.profileOverflow,
    };
  }

  /**
   * Per-pass helper. Allocates two query slots, records the label, returns a
   * compute pass configured to write the begin/end timestamps. On unsupported
   * adapters or when profiling is off, behaves exactly like `beginComputePass()`
   * with zero additional cost.
   */
  private beginPass(encoder: GPUCommandEncoder, label: string): GPUComputePassEncoder {
    if (this.cpuProfileActive && this.cpuProfileCapturing) this.cpuPassCountThisForward++;
    if (!this.profileActive || !this.profileCapturing || !this.profileQuerySet) {
      return encoder.beginComputePass();
    }
    const slot = this.profileSlotCursor;
    if (slot + 2 > this.profileCapability.querySlots) {
      this.profileOverflow = true;
      return encoder.beginComputePass();
    }
    this.profileSlotCursor = slot + 2;
    this.profilePassLabels.push(label);
    return encoder.beginComputePass({
      timestampWrites: {
        querySet: this.profileQuerySet,
        beginningOfPassWriteIndex: slot,
        endOfPassWriteIndex: slot + 1,
      },
    });
  }

  /**
   * Must be called on the encoder after all passes have ended and before the
   * encoder is submitted. Appends `resolveQuerySet` + `copyBufferToBuffer` so
   * timestamps land in the mappable staging buffer. No-op when profiling is off.
   * Returns a flag the caller stashes to know whether to await timestamps after submit.
   */
  private appendProfileResolve(encoder: GPUCommandEncoder): boolean {
    if (
      !this.profileActive ||
      !this.profileCapturing ||
      !this.profileQuerySet ||
      !this.profileResolveBuf ||
      !this.profileStagingBuf
    ) {
      return false;
    }
    const slotsUsed = this.profileSlotCursor;
    if (slotsUsed === 0) return false;
    encoder.resolveQuerySet(this.profileQuerySet, 0, slotsUsed, this.profileResolveBuf, 0);
    encoder.copyBufferToBuffer(this.profileResolveBuf, 0, this.profileStagingBuf, 0, slotsUsed * 8);
    return true;
  }

  /** Map the staging buffer, decode timestamp pairs, accumulate per-label samples, then unmap + reset cursor. */
  private async collectProfileSamples(): Promise<void> {
    const staging = this.profileStagingBuf!;
    const labels = this.profilePassLabels;
    const slotsUsed = this.profileSlotCursor;
    // Reset cursor/labels eagerly so the next forward pass can start encoding
    // immediately after the map finishes. Use local copies for the decode.
    this.profileSlotCursor = 0;
    this.profilePassLabels = [];
    if (slotsUsed === 0) return;
    try {
      await staging.mapAsync(GPUMapMode.READ, 0, slotsUsed * 8);
    } catch (e) {
      // If mapAsync races a device-lost or cancellation, bail without crashing the hot path.
      console.warn(`profile staging map failed: ${e}`);
      return;
    }
    const ts = new BigInt64Array(staging.getMappedRange(0, slotsUsed * 8).slice(0));
    staging.unmap();
    let forwardTotal = 0;
    for (let i = 0; i < labels.length; i++) {
      const begin = ts[i * 2];
      const end = ts[i * 2 + 1];
      // GPUQuerySet can emit zero if a pass was empty; skip those.
      if (end <= begin) continue;
      const delta = Number(end - begin);
      forwardTotal += delta;
      const label = labels[i];
      let arr = this.profileSamples.get(label);
      if (!arr) {
        arr = [];
        this.profileSamples.set(label, arr);
      }
      arr.push(delta);
    }
    if (forwardTotal > 0) this.profileForwardTotals_ns.push(forwardTotal);
  }

  dispose(): void {
    // Destroy all GPU buffers
    const destroyBuffer = (buf: GPUBuffer | null) => { if (buf) buf.destroy(); };
    if (this.profileResolveBuf) destroyBuffer(this.profileResolveBuf);
    if (this.profileStagingBuf) destroyBuffer(this.profileStagingBuf);
    if (this.profileQuerySet) this.profileQuerySet.destroy();

    destroyBuffer(this.modelBuffers?.tokenEmbed);
    destroyBuffer(this.modelBuffers?.finalNorm);
    if (this.modelBuffers?.layers) {
      for (const layer of this.modelBuffers.layers) {
        for (const buf of Object.values(layer)) {
          destroyBuffer(buf);
        }
      }
    }

    if (this.workBuffers) {
      for (const val of Object.values(this.workBuffers)) {
        // `argmaxReadbackPool` is a tuple [GPUBuffer, GPUBuffer]; every
        // other entry is a single buffer.
        if (Array.isArray(val)) {
          for (const buf of val) destroyBuffer(buf);
        } else {
          destroyBuffer(val as GPUBuffer);
        }
      }
    }

    if (this.kvCaches) {
      for (const kv of this.kvCaches) {
        destroyBuffer(kv.k);
        destroyBuffer(kv.v);
      }
    }

    if (this.uniformBuffers) {
      for (const val of Object.values(this.uniformBuffers)) {
        if (Array.isArray(val)) {
          for (const buf of val) destroyBuffer(buf);
        } else {
          destroyBuffer(val as GPUBuffer);
        }
      }
    }

    this.device?.destroy();
  }
}
