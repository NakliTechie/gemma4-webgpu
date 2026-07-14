// Unlimited-OCR end-to-end P1 smoke — the whole vision→decoder chain in one tab:
//
//   canvas document → preprocess (1024², mean/std 0.5, CHW)
//     → DeepEncoder ONNX (ORT-web, fp32 export) → [256, 1280] vision embeds
//     → splice: 16 rows × (16 patches + image_newline) + view_seperator = 273
//     → engine prefill: [BOS] + 273 embeds (prefillEmbedsForCapture) + prompt ids
//     → greedy decode on the custom-WGSL MoE decoder.
//
// Agent instrument: everything on `window.ocr`. ocr.run() does the full chain.
// Token ids for the text parts are precomputed with the HF tokenizer (the
// engine's BPE handles them too, but ids are pinned here so the demo can't
// drift from the reference tokenization):
//   BOS = 0 · "\nFree OCR." = [201, 21431, 126041, 16] · EOS = 1
import { createGemmaEngine } from '../src/index.js';
import type { GemmaEngine } from '../src/index.js';
import { loadReferenceTensors } from '../src/diagnostics/index.js';

const MODEL_URL = '/models/Unlimited-OCR-Q4_K_M.gguf';
const VISION_URL = '/models/onnx/deepencoder_fp32.onnx';
const EXTRAS_URL = '/models/onnx/deepencoder_extras.npz';
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.webgpu.min.mjs';

const BOS = 0;
const EOS = 1;
const PROMPT_IDS = [201, 21431, 126041, 16]; // "\nFree OCR."
const H = 1280;
const GRID = 16; // 16×16 vision tokens per 1024² view

const statusEl = () => document.getElementById('status')!;
const outEl = () => document.getElementById('out')!;
const say = (s: string) => { statusEl().textContent = s; console.log('[ocr]', s); };

/** Draw a synthetic document worth OCRing. */
function drawDocument(): void {
  const cv = document.getElementById('doc') as HTMLCanvasElement;
  const g = cv.getContext('2d')!;
  g.fillStyle = '#ffffff';
  g.fillRect(0, 0, 1024, 1024);
  g.fillStyle = '#000000';
  g.font = 'bold 56px Georgia';
  g.fillText('Quarterly Report', 64, 120);
  g.font = '34px Georgia';
  g.fillText('Revenue grew 18% year over year.', 64, 220);
  g.fillText('Operating costs fell by 4%.', 64, 280);
  g.fillText('The board approved the 2027 plan.', 64, 340);
  g.font = 'italic 28px Georgia';
  g.fillText('Prepared by the finance team.', 64, 430);
}

/** Canvas → normalized CHW f32 (mean 0.5 / std 0.5, matching the mmproj metadata). */
function preprocess(): Float32Array {
  const cv = document.getElementById('doc') as HTMLCanvasElement;
  const g = cv.getContext('2d')!;
  const { data } = g.getImageData(0, 0, 1024, 1024);
  const out = new Float32Array(3 * 1024 * 1024);
  const plane = 1024 * 1024;
  for (let i = 0; i < plane; i++) {
    out[i] = (data[i * 4] / 255 - 0.5) / 0.5;
    out[plane + i] = (data[i * 4 + 1] / 255 - 0.5) / 0.5;
    out[2 * plane + i] = (data[i * 4 + 2] / 255 - 0.5) / 0.5;
  }
  return out;
}

/** [256,1280] patches + newline/seperator → [273,1280] spliced sequence. */
function spliceVision(patches: Float32Array, newline: Float32Array, seperator: Float32Array): Float32Array {
  const rows = GRID * (GRID + 1) + 1; // 273
  const out = new Float32Array(rows * H);
  let o = 0;
  for (let r = 0; r < GRID; r++) {
    out.set(patches.subarray(r * GRID * H, (r + 1) * GRID * H), o); o += GRID * H;
    out.set(newline, o); o += H;
  }
  out.set(seperator, o);
  return out;
}

interface Ocr {
  engine: GemmaEngine | null;
  visionMs: number | null;
  result: string | null;
  run: (opts?: { maxTokens?: number }) => Promise<string>;
}

const ocr: Ocr = {
  engine: null,
  visionMs: null,
  result: null,

  async run(opts = {}) {
    const maxTokens = opts.maxTokens ?? 220;
    drawDocument();

    say('loading DeepEncoder ONNX (ORT-web, WebGPU EP)…');
    const ort = await import(/* @vite-ignore */ ORT_URL);
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
    const sess = await ort.InferenceSession.create(VISION_URL, {
      executionProviders: ['webgpu', 'wasm'],
    });

    say('vision forward…');
    const t0 = performance.now();
    const feeds = { pixel_values: new ort.Tensor('float32', preprocess(), [1, 3, 1024, 1024]) };
    const results = await sess.run(feeds);
    const embeds = results.vision_embeds.data as Float32Array; // [1,256,1280]
    this.visionMs = Math.round(performance.now() - t0);
    say(`vision done in ${this.visionMs} ms — ‖v‖=${Math.hypot(...embeds.slice(0, 4)).toFixed(3)}… loading extras`);

    const { tensors } = await loadReferenceTensors(EXTRAS_URL);
    const newline = tensors['image_newline'] as Float32Array;
    const seperator = tensors['view_seperator'] as Float32Array;
    const visionSeq = spliceVision(embeds, newline, seperator);

    if (!this.engine) {
      say('loading decoder GGUF…');
      this.engine = await createGemmaEngine({
        model: MODEL_URL,
        weightQuant: 'q4k',
        contextLength: 2048,
        onProgress: (p) => say(`decoder: ${p.status} ${p.total ? ((p.loaded / p.total) * 100).toFixed(0) + '%' : ''}`),
      });
    }
    const eng = this.engine;

    say('prefill: BOS + 273 vision embeds + prompt…');
    eng.resetKVForCapture();
    await eng.prefillForCapture([BOS], 0);
    await eng.prefillEmbedsForCapture(visionSeq, 1);
    const textStart = 1 + visionSeq.length / H;
    await eng.prefillForCapture(PROMPT_IDS.slice(0, -1), textStart);

    say('greedy decode…');
    let tok = PROMPT_IDS[PROMPT_IDS.length - 1];
    let pos = textStart + PROMPT_IDS.length - 1;
    const ids: number[] = [];
    const tDec = performance.now();
    for (let i = 0; i < maxTokens; i++) {
      const logits = await eng.captureHidden(tok, pos, { kind: 'logits' });
      let best = 0;
      for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
      if (best === EOS) break;
      ids.push(best); tok = best; pos++;
      if (i % 16 === 0) say(`decoding… ${i} tokens`);
    }
    const decMs = performance.now() - tDec;
    this.result = eng.decodeTokens(ids);
    outEl().textContent = this.result;
    say(`done — vision ${this.visionMs} ms · ${ids.length} tokens in ${(decMs / 1000).toFixed(1)} s (${(ids.length / (decMs / 1000)).toFixed(1)} tok/s incl. per-token readback)`);
    return this.result;
  },
};

declare global { interface Window { ocr: Ocr } }
window.ocr = ocr;
drawDocument();
