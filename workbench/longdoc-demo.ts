// P2 long-doc demo + Tier-2/Tier-3 verification harness.
//
// The R-SWA headline, made falsifiable in one number: the engine is
// initialized with contextLength = 1024 KV slots, the 3-page prefix alone is
// ~824 positions, and the run decodes hundreds of tokens — TOTAL positions
// sail past the allocated window. Without the ring that's an out-of-bounds
// cache write; with it, generated tokens recycle 128 slots and KV memory is
// CONSTANT in output length.
//
//   window.longdoc.run()          — full chain vs the HF bf16 ring control
//   window.longdoc.teacherForce() — argmax agreement past the wrap, teacher-forced
import { createGemmaEngine } from '../src/index.js';
import type { GemmaEngine } from '../src/index.js';

const MODEL_URL = '/models/Unlimited-OCR-Q4_K_M.gguf';
const VISION_URL = '/models/onnx/deepencoder_fp32.onnx';
const EXTRAS_URL = '/models/onnx/deepencoder_extras.npz';
const CONTROL_URL = '/models/longdoc/hf_ring_control.json';
const PAGE_URLS = ['/models/longdoc/page_1.png', '/models/longdoc/page_2.png', '/models/longdoc/page_3.png'];
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.webgpu.min.mjs';

const BOS = 0, EOS = 1, H = 1280, GRID = 16;
const CTX = 1024; // deliberately smaller than prefix+output — the ring makes it enough

const say = (s: string) => { document.getElementById('status')!.textContent = s; console.log('[longdoc]', s); };
const metrics = (s: string) => { document.getElementById('metrics')!.textContent = s; };

async function drawPage(idx: number, url: string): Promise<void> {
  const img = new Image();
  img.src = url;
  await img.decode();
  const cv = document.getElementById('p' + idx) as HTMLCanvasElement;
  cv.getContext('2d')!.drawImage(img, 0, 0, 1024, 1024);
}

function preprocess(idx: number): Float32Array {
  const cv = document.getElementById('p' + idx) as HTMLCanvasElement;
  const { data } = cv.getContext('2d')!.getImageData(0, 0, 1024, 1024);
  const out = new Float32Array(3 * 1024 * 1024);
  const plane = 1024 * 1024;
  for (let i = 0; i < plane; i++) {
    out[i] = (data[i * 4] / 255 - 0.5) / 0.5;
    out[plane + i] = (data[i * 4 + 1] / 255 - 0.5) / 0.5;
    out[2 * plane + i] = (data[i * 4 + 2] / 255 - 0.5) / 0.5;
  }
  return out;
}

function splicePage(patches: Float32Array, newline: Float32Array, seperator: Float32Array): Float32Array {
  const out = new Float32Array((GRID * (GRID + 1) + 1) * H);
  let o = 0;
  for (let r = 0; r < GRID; r++) {
    out.set(patches.subarray(r * GRID * H, (r + 1) * GRID * H), o); o += GRID * H;
    out.set(newline, o); o += H;
  }
  out.set(seperator, o);
  return out;
}

interface Ctl {
  prompt_ids: number[]; n_img_tokens: number; seq_prefix_len: number;
  ring_ids: number[]; ring_text: string; plain_ids: number[]; first_divergence: number;
}

interface LongDoc {
  engine: GemmaEngine | null;
  ctl: Ctl | null;
  visionSeq: Float32Array | null;
  P: number;
  result: { ids: number[]; text: string } | null;
  prepare: () => Promise<void>;
  run: (opts?: { maxTokens?: number }) => Promise<Record<string, unknown>>;
  teacherForce: (steps?: number[]) => Promise<Record<string, unknown>>;
}

const longdoc: LongDoc = {
  engine: null, ctl: null, visionSeq: null, P: 0, result: null,

  /** Vision for all pages + control + engine load. Idempotent. */
  async prepare() {
    if (this.visionSeq && this.engine && this.ctl) return;
    this.ctl = await (await fetch(CONTROL_URL)).json() as Ctl;
    for (let i = 0; i < PAGE_URLS.length; i++) await drawPage(i, PAGE_URLS[i]);

    say('vision: loading DeepEncoder ONNX…');
    const ort = await import(/* @vite-ignore */ ORT_URL);
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
    const sess = await ort.InferenceSession.create(VISION_URL, { executionProviders: ['webgpu', 'wasm'] });
    const { loadReferenceTensors } = await import('../src/diagnostics/index.js');
    const { tensors } = await loadReferenceTensors(EXTRAS_URL);
    const newline = tensors['image_newline'] as Float32Array;
    const seperator = tensors['view_seperator'] as Float32Array;

    const blocks: Float32Array[] = [];
    for (let i = 0; i < PAGE_URLS.length; i++) {
      const t0 = performance.now();
      const r = await sess.run({ pixel_values: new ort.Tensor('float32', preprocess(i), [1, 3, 1024, 1024]) });
      blocks.push(splicePage(r.vision_embeds.data as Float32Array, newline, seperator));
      say(`vision: page ${i + 1}/${PAGE_URLS.length} in ${(performance.now() - t0).toFixed(0)} ms`);
    }
    const total = blocks.reduce((a, b) => a + b.length, 0);
    const seq = new Float32Array(total);
    let o = 0; for (const b of blocks) { seq.set(b, o); o += b.length; }
    this.visionSeq = seq;

    if (!this.engine) {
      say(`decoder: loading (contextLength=${CTX} — smaller than prefix+output on purpose)…`);
      this.engine = await createGemmaEngine({
        model: MODEL_URL, weightQuant: 'q4k', contextLength: CTX,
        onProgress: (p) => say(`decoder: ${p.status}`),
      });
    }
    this.P = 1 + this.visionSeq.length / H + this.ctl.prompt_ids.length;
  },

  /** Free-running greedy with the ring vs the HF bf16 ring control. */
  async run(opts = {}) {
    await this.prepare();
    const eng = this.engine!, ctl = this.ctl!;
    const maxTokens = opts.maxTokens ?? 480;
    const nImg = this.visionSeq!.length / H;
    const P = this.P;

    say(`prefill: BOS + ${nImg} vision embeds + ${ctl.prompt_ids.length} prompt ids (P=${P}, ctx=${CTX})…`);
    eng.resetKVForCapture();
    await eng.prefillForCapture([BOS], 0);
    await eng.prefillEmbedsForCapture(this.visionSeq!, 1);
    await eng.prefillForCapture(ctl.prompt_ids.slice(0, -1), 1 + nImg);
    eng.beginRingDecode(P);

    let tok = ctl.prompt_ids[ctl.prompt_ids.length - 1], pos = P - 1;
    const ids: number[] = [];
    const t0 = performance.now();
    let maxKvLen = 0;
    for (let i = 0; i < maxTokens; i++) {
      const logits = await eng.captureHidden(tok, pos, { kind: 'logits' });
      let best = 0;
      for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
      maxKvLen = Math.min(pos + 1, P + 128);
      if (best === EOS) break;
      ids.push(best); tok = best; pos++;
      if (i % 32 === 0) say(`decode ${i}: pos=${pos} (past ctx ${CTX}? ${pos >= CTX}) · KV entries=${maxKvLen}`);
    }
    const secs = (performance.now() - t0) / 1000;
    const text = eng.decodeTokens(ids);
    this.result = { ids, text };
    document.getElementById('out')!.textContent = text;

    let firstDivVsHF = -1;
    const n = Math.min(ids.length, ctl.ring_ids.length);
    for (let i = 0; i < n; i++) if (ids[i] !== ctl.ring_ids[i]) { firstDivVsHF = i; break; }
    const finalPos = P + ids.length;
    const summary = {
      generated: ids.length, seconds: +secs.toFixed(1), tps: +(ids.length / secs).toFixed(1),
      P, ctx: CTX, finalPos, exceededCtx: finalPos > CTX,
      kvEntriesPlateau: maxKvLen, expectedPlateau: P + 128,
      firstDivVsHFRing: firstDivVsHF, hfRingLen: ctl.ring_ids.length,
      textHead: text.slice(0, 200),
    };
    metrics(JSON.stringify(summary, null, 1));
    say('run done');
    return summary;
  },

  /** Teacher-forced Tier-2: feed the HF ring token stream, compare engine argmax
   *  at each step PAST THE WRAP — isolates ring correctness from greedy drift. */
  async teacherForce(steps) {
    await this.prepare();
    const eng = this.engine!, ctl = this.ctl!;
    const P = this.P;
    const hf = ctl.ring_ids;
    const checkFrom = 128;
    const checkSteps = steps ?? Array.from({ length: Math.min(80, hf.length - checkFrom - 1) }, (_, k) => checkFrom + k);

    say('teacherForce: re-prefilling…');
    const nImg = this.visionSeq!.length / H;
    eng.resetKVForCapture();
    await eng.prefillForCapture([BOS], 0);
    await eng.prefillEmbedsForCapture(this.visionSeq!, 1);
    await eng.prefillForCapture(ctl.prompt_ids.slice(0, -1), 1 + nImg);
    eng.beginRingDecode(P);

    let tok = ctl.prompt_ids[ctl.prompt_ids.length - 1], pos = P - 1;
    let agree = 0, total = 0;
    const misses: Array<{ step: number; ours: number; hf: number }> = [];
    const last = Math.max(...checkSteps);
    for (let i = 0; i <= last && i < hf.length; i++) {
      const logits = await eng.captureHidden(tok, pos, { kind: 'logits' });
      if (checkSteps.includes(i)) {
        let best = 0;
        for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
        total++;
        if (best === hf[i]) agree++;
        else if (misses.length < 10) misses.push({ step: i, ours: best, hf: hf[i] });
      }
      tok = hf[i]; pos++;   // teacher-force the HF ring stream
      if (i % 40 === 0) say(`teacherForce step ${i}/${last}`);
    }
    const summary = { checkedPastWrap: total, agree, agreeRate: +(agree / total).toFixed(4), misses };
    metrics(JSON.stringify(summary, null, 1));
    say('teacherForce done');
    return summary;
  },
};

declare global { interface Window { longdoc: LongDoc } }
window.longdoc = longdoc;
say('idle — window.longdoc.run() (needs hf_ring_control.json from the python side)');
