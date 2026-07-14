// P2 showpiece: a REAL multi-page PDF parsed one-shot, fully client-side —
// pdf.js renders pages → DeepEncoder ONNX (ORT-web WebGPU, in-tab) → ring-KV
// MoE decode with constant memory. No precomputed anything: the only inputs
// are the .pdf bytes and the two model files.
//
// Order matters for renderer memory: pdf.js render → ALL vision forwards →
// session.release() → decoder load → decode. (Vision + decoder coexisting
// peaked past the tab budget on earlier runs.)
import { createGemmaEngine } from '../src/index.js';
import type { GemmaEngine } from '../src/index.js';
import { loadReferenceTensors } from '../src/diagnostics/index.js';

const PDF_URL = '/models/longdoc/annual_letter.pdf';
const MODEL_URL = '/models/Unlimited-OCR-Q4_K_M.gguf';
const VISION_URL = '/models/onnx/deepencoder_fp32.onnx';
const EXTRAS_URL = '/models/onnx/deepencoder_extras.npz';
const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.webgpu.min.mjs';
const PDFJS_URL = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.8.69/build/pdf.min.mjs';
const PDFJS_WORKER = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.8.69/build/pdf.worker.min.mjs';

const BOS = 0, EOS = 1, H = 1280, GRID = 16;
const PROMPT_IDS = [37460, 4366, 76466, 16]; // "Multi page parsing."
const CTX = 2048;

const say = (s: string) => { document.getElementById('status')!.textContent = s; console.log('[pdf]', s); };

function pageToTensor(cv: HTMLCanvasElement): Float32Array {
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

interface PdfDemo {
  engine: GemmaEngine | null;
  result: string | null;
  run: (opts?: { maxTokens?: number }) => Promise<Record<string, unknown>>;
}

const pdfdemo: PdfDemo = {
  engine: null, result: null,

  async run(opts = {}) {
    const maxTokens = opts.maxTokens ?? 640;
    const tAll = performance.now();

    say('pdf.js: importing module…');
    const pdfjs = await import(/* @vite-ignore */ PDFJS_URL);
    // Workers can't be constructed from a cross-origin URL — fetch the worker
    // script and hand pdf.js a same-origin blob URL (the raw CDN workerSrc
    // hangs getDocument() silently).
    say('pdf.js: fetching worker…');
    const workerSrc = await (await fetch(PDFJS_WORKER)).text();
    const workerUrl = URL.createObjectURL(new Blob([workerSrc], { type: 'text/javascript' }));
    // pdf.worker.min.mjs is an ES MODULE — pdf.js spawns workerSrc as a
    // CLASSIC worker, which dies on module syntax and stalls the handshake
    // forever (the previous silent hang). Construct the module worker
    // ourselves and hand over the port.
    pdfjs.GlobalWorkerOptions.workerPort = new Worker(workerUrl, { type: 'module' });
    say('pdf.js: opening document…');
    const pdf = await pdfjs.getDocument(PDF_URL).promise;
    const pagesEl = document.getElementById('pages')!;
    pagesEl.textContent = '';
    const canvases: HTMLCanvasElement[] = [];
    for (let p = 1; p <= pdf.numPages; p++) {
      const page = await pdf.getPage(p);
      const base = page.getViewport({ scale: 1 });
      const scale = 1024 / Math.max(base.width, base.height);
      const vp = page.getViewport({ scale });
      const cv = document.createElement('canvas');
      cv.width = 1024; cv.height = 1024;
      const g = cv.getContext('2d')!;
      g.fillStyle = '#fff'; g.fillRect(0, 0, 1024, 1024); // pad to square, white
      await page.render({ canvasContext: g, viewport: vp }).promise;
      pagesEl.appendChild(cv);
      canvases.push(cv);
      say(`pdf.js: page ${p}/${pdf.numPages} rendered`);
    }

    say('vision: DeepEncoder ONNX (in-tab, WebGPU EP)…');
    const ort = await import(/* @vite-ignore */ ORT_URL);
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
    const sess = await ort.InferenceSession.create(VISION_URL, { executionProviders: ['webgpu', 'wasm'] });
    const { tensors } = await loadReferenceTensors(EXTRAS_URL);
    const newline = tensors['image_newline'] as Float32Array;
    const seperator = tensors['view_seperator'] as Float32Array;
    const blocks: Float32Array[] = [];
    let visionMs = 0;
    for (let i = 0; i < canvases.length; i++) {
      const t0 = performance.now();
      const r = await sess.run({ pixel_values: new ort.Tensor('float32', pageToTensor(canvases[i]), [1, 3, 1024, 1024]) });
      visionMs += performance.now() - t0;
      blocks.push(splicePage(r.vision_embeds.data as Float32Array, newline, seperator));
      say(`vision: page ${i + 1}/${canvases.length} done`);
    }
    await sess.release();
    const visionSeq = new Float32Array(blocks.reduce((a, b) => a + b.length, 0));
    { let o = 0; for (const b of blocks) { visionSeq.set(b, o); o += b.length; } }

    if (!this.engine) {
      say('decoder: loading…');
      this.engine = await createGemmaEngine({
        model: MODEL_URL, weightQuant: 'q4k', contextLength: CTX,
        onProgress: (p) => say(`decoder: ${p.status}`),
      });
    }
    const eng = this.engine;

    const nImg = visionSeq.length / H;
    const P = 1 + nImg + PROMPT_IDS.length;
    say(`prefill: BOS + ${nImg} vision + prompt (P=${P})…`);
    eng.resetKVForCapture();
    await eng.prefillForCapture([BOS], 0);
    await eng.prefillEmbedsForCapture(visionSeq, 1);
    await eng.prefillForCapture(PROMPT_IDS.slice(0, -1), 1 + nImg);
    eng.beginRingDecode(P);

    let tok = PROMPT_IDS[PROMPT_IDS.length - 1], pos = P - 1;
    const ids: number[] = [];
    const tDec = performance.now();
    for (let i = 0; i < maxTokens; i++) {
      const logits = await eng.captureHidden(tok, pos, { kind: 'logits' });
      let best = 0;
      for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
      if (best === EOS) break;
      ids.push(best); tok = best; pos++;
      if (i % 32 === 0) say(`decode ${i} (pos ${pos}, KV ≤ ${P + 128})…`);
    }
    const decS = (performance.now() - tDec) / 1000;
    this.result = eng.decodeTokens(ids);
    document.getElementById('out')!.textContent = this.result;
    const summary = {
      pdfPages: pdf.numPages, generated: ids.length, hitEos: ids.length < maxTokens,
      visionSecs: +(visionMs / 1000).toFixed(1), decodeSecs: +decS.toFixed(1),
      tps: +(ids.length / decS).toFixed(1), totalSecs: +((performance.now() - tAll) / 1000).toFixed(1),
      P, kvCap: P + 128, textHead: this.result.slice(0, 160),
    };
    document.getElementById('metrics')!.textContent = JSON.stringify(summary, null, 1);
    say('done');
    return summary;
  },
};

declare global { interface Window { pdfdemo: PdfDemo } }
window.pdfdemo = pdfdemo;
say('idle — window.pdfdemo.run()');
