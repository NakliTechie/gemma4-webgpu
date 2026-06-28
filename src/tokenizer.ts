import type { GGUFParsed } from './types.js';
import { kvArray } from './gguf.js';

const SPECIAL_TOKENS: Record<string, number> = {
  '<start_of_turn>': 105,
  '<end_of_turn>': 106,
  '<eos>': 1,
  '<bos>': 2,
};

const FUNC_TOKEN_NAMES = [
  '<start_function_declaration>', '<end_function_declaration>',
  '<start_function_call>', '<end_function_call>',
  '<start_function_response>', '<end_function_response>',
  '<escape>',
];

// Qwen3 / GPT-2 byte-level BPE specials (looked up in the vocab if present).
const BPE_SPECIAL_NAMES = [
  '<|im_start|>', '<|im_end|>', '<|endoftext|>',
  '<|object_ref_start|>', '<|object_ref_end|>',
  '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>',
  '<|vision_start|>', '<|vision_end|>', '<|vision_pad|>',
  '<|image_pad|>', '<|video_pad|>',
  '<tool_call>', '</tool_call>', '<|fim_prefix|>', '<|fim_middle|>',
  '<|fim_suffix|>', '<|fim_pad|>', '<|repo_name|>', '<|file_sep|>',
];

// GPT-2 reversible byte↔unicode map. Printable ranges map to themselves; the
// remaining bytes (incl. space 0x20 → 'Ġ' U+0120) map to U+0100+. Lets a
// byte-level BPE vocab (where " Paris" is stored as "ĠParis") round-trip.
function bytesToUnicode(): Map<number, string> {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const m = new Map<number, string>();
  for (let i = 0; i < bs.length; i++) m.set(bs[i], String.fromCodePoint(cs[i]));
  return m;
}

export class Tokenizer {
  vocab: string[] = [];
  vocabByLength: [number, string][] = [];
  tokenByText: Map<string, number> = new Map();
  maxTokenLen: number = 0;
  specialTokens: Record<string, number> = { ...SPECIAL_TOKENS };
  funcTokens: Record<string, number> = {};
  private specialPatternRegex: RegExp = /\\<start_of_turn\\>|\\<end_of_turn\\>|\\<eos\\>|\\<bos\\>/g;

  // Byte-level BPE state (Qwen3 / gpt2). `mode` is chosen at extract time:
  // 'spm' = Gemma SentencePiece (▁ marker), 'bpe' = GPT-2 byte-level (merges).
  private mode: 'spm' | 'bpe' = 'spm';
  private byteEncoder: Map<number, string> = new Map();
  private byteDecoder: Map<string, number> = new Map();
  private bpeRanks: Map<string, number> = new Map();
  private addBos = false;
  // Qwen2/gpt2 pre-tokenizer split (\p{L}/\p{N} need the `u` flag).
  private bpePat = /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

  extractFromGGUF(gguf: GGUFParsed): void {
    const tokens = kvArray(gguf, 'tokenizer.ggml.tokens') as string[] | null;
    if (!tokens) throw new Error('No tokenizer found in GGUF metadata');
    this.vocab = tokens;
    this.vocabByLength = [];
    for (let i = 0; i < this.vocab.length; i++) {
      if (this.vocab[i] && this.vocab[i].length > 0) this.vocabByLength.push([i, this.vocab[i]]);
    }
    this.buildTokenIndex();

    // Byte-level BPE (Qwen3 etc.) is signalled by the presence of merge rules,
    // which SentencePiece (Gemma) GGUFs don't carry.
    const merges = kvArray(gguf, 'tokenizer.ggml.merges') as string[] | null;
    if (merges && merges.length > 0) {
      this.mode = 'bpe';
      this.byteEncoder = bytesToUnicode();
      for (const [b, c] of this.byteEncoder) this.byteDecoder.set(c, b);
      for (let i = 0; i < merges.length; i++) {
        const sp = merges[i].indexOf(' ');
        if (sp <= 0) continue;
        this.bpeRanks.set(merges[i].slice(0, sp) + '' + merges[i].slice(sp + 1), i);
      }
      const addBos = gguf.kv.get('tokenizer.ggml.add_bos_token')?.value;
      this.addBos = addBos === true;
      this.initBpeSpecials();
    } else {
      this.mode = 'spm';
      this.initFunctionTokens();
    }
  }

  private buildTokenIndex(): void {
    this.tokenByText = new Map();
    this.maxTokenLen = 0;
    for (let i = 0; i < this.vocab.length; i++) {
      const token = this.vocab[i];
      if (token && token.length > 0) {
        if (!this.tokenByText.has(token)) this.tokenByText.set(token, i);
        if (token.length > this.maxTokenLen) this.maxTokenLen = token.length;
      }
    }
  }

  private initFunctionTokens(): void {
    for (const name of FUNC_TOKEN_NAMES) {
      const id = this.tokenByText.get(name);
      if (id !== undefined) { this.funcTokens[name] = id; this.specialTokens[name] = id; }
    }
    this.rebuildSpecialPattern();
  }

  private initBpeSpecials(): void {
    this.specialTokens = {};
    for (const name of BPE_SPECIAL_NAMES) {
      const id = this.tokenByText.get(name);
      if (id !== undefined) this.specialTokens[name] = id;
    }
    this.rebuildSpecialPattern();
  }

  private rebuildSpecialPattern(): void {
    const allTokenNames = Object.keys(this.specialTokens);
    if (allTokenNames.length === 0) { this.specialPatternRegex = /(?!)/g; return; }
    allTokenNames.sort((a, b) => b.length - a.length);
    const specialPatternStr = allTokenNames.map((t) => t.replace(/[.*+?^${}()|[\]\\<>]/g, '\\$&')).join('|');
    this.specialPatternRegex = new RegExp(specialPatternStr, 'g');
  }

  // ── SentencePiece (Gemma) ──────────────────────────────────────────────
  private encodeSegment(text: string, addPrefix: boolean = true): number[] {
    const tokens: number[] = [];
    let remaining = text.replace(/ /g, '▁');
    if (addPrefix) remaining = '▁' + remaining;
    while (remaining.length > 0) {
      let bestLen = 0, bestId = -1;
      const tryLen = Math.min(remaining.length, this.maxTokenLen);
      for (let len = tryLen; len >= 1; len--) {
        const id = this.tokenByText.get(remaining.substring(0, len));
        if (id !== undefined) { bestLen = len; bestId = id; break; }
      }
      if (bestLen === 0) remaining = remaining.slice(1);
      else { tokens.push(bestId); remaining = remaining.slice(bestLen); }
    }
    return tokens;
  }

  // ── GPT-2 byte-level BPE (Qwen3) ───────────────────────────────────────
  private bpe(word: string): string[] {
    let parts = Array.from(word);
    if (parts.length <= 1) return parts;
    for (;;) {
      let minRank = Infinity, minI = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.bpeRanks.get(parts[i] + '' + parts[i + 1]);
        if (r !== undefined && r < minRank) { minRank = r; minI = i; }
      }
      if (minI < 0) break;
      parts = parts.slice(0, minI).concat(parts[minI] + parts[minI + 1], parts.slice(minI + 2));
    }
    return parts;
  }

  private encodeBpeChunk(text: string): number[] {
    const out: number[] = [];
    const enc = new TextEncoder();
    let m: RegExpExecArray | null;
    this.bpePat.lastIndex = 0;
    while ((m = this.bpePat.exec(text)) !== null) {
      const bytes = enc.encode(m[0]);
      let s = '';
      for (const b of bytes) s += this.byteEncoder.get(b)!;
      for (const sym of this.bpe(s)) {
        const id = this.tokenByText.get(sym);
        if (id !== undefined) out.push(id);
        else for (const ch of sym) { const bid = this.tokenByText.get(ch); if (bid !== undefined) out.push(bid); }
      }
    }
    return out;
  }

  private encodeBpe(text: string): number[] {
    const tokens: number[] = [];
    if (this.addBos) { const bos = this.specialTokens['<|endoftext|>']; if (bos !== undefined) tokens.push(bos); }
    const special = new RegExp(this.specialPatternRegex.source, 'g');
    let lastIdx = 0, match: RegExpExecArray | null;
    while ((match = special.exec(text)) !== null) {
      if (match.index > lastIdx) tokens.push(...this.encodeBpeChunk(text.slice(lastIdx, match.index)));
      tokens.push(this.specialTokens[match[0]]);
      lastIdx = match.index + match[0].length;
    }
    if (lastIdx < text.length) tokens.push(...this.encodeBpeChunk(text.slice(lastIdx)));
    return tokens;
  }

  encode(text: string): number[] {
    if (this.mode === 'bpe') return this.encodeBpe(text);
    const tokens = [2]; // BOS (Gemma)
    const specialPattern = new RegExp(this.specialPatternRegex.source, 'g');
    let lastIdx = 0, match: RegExpExecArray | null, afterSpecial = false;
    while ((match = specialPattern.exec(text)) !== null) {
      const before = text.slice(lastIdx, match.index);
      if (before.length > 0) tokens.push(...this.encodeSegment(before, !afterSpecial));
      tokens.push(this.specialTokens[match[0]]);
      lastIdx = match.index + match[0].length;
      afterSpecial = true;
    }
    const remaining = text.slice(lastIdx);
    if (remaining.length > 0) tokens.push(...this.encodeSegment(remaining, !afterSpecial));
    return tokens;
  }

  decodeToken(tokenId: number): string {
    if (tokenId < this.vocab.length && this.vocab[tokenId]) {
      if (this.mode === 'bpe') return this.decodeTokens([tokenId]);
      return this.vocab[tokenId].replace(/▁/g, ' ');
    }
    return `<unk:${tokenId}>`;
  }

  decodeTokens(tokenIds: number[]): string {
    if (this.mode === 'bpe') {
      const bytes: number[] = [];
      for (const id of tokenIds) {
        const s = id < this.vocab.length ? this.vocab[id] : undefined;
        if (!s) continue;
        for (const ch of s) {
          const b = this.byteDecoder.get(ch);
          if (b !== undefined) bytes.push(b);
          else for (const u of new TextEncoder().encode(ch)) bytes.push(u);
        }
      }
      return new TextDecoder().decode(Uint8Array.from(bytes));
    }
    let text = '';
    for (const id of tokenIds) text += this.decodeToken(id);
    return text;
  }
}
