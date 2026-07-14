// Cross-check the src/gguf.ts Q5_0 dequant against the python `gguf` reference.
//
//   node --experimental-strip-types scripts/check_q5_0.mjs <model.gguf> <tensor> [nElems]
//
// Imports src/gguf.ts directly (Node ≥22.6 type stripping; the only import in
// gguf.ts is type-only, so no further resolution is needed). Reads the header
// from a 64 MB prefix, then reads ONLY the target tensor's bytes and dequants.
// Prints one value per line for diffing against scripts/check_q5_0.py.
import { openSync, readSync } from 'node:fs';
import { GGUFParser, tensorByteSize } from '../src/gguf.ts';

const [file, tensorName, nStr] = process.argv.slice(2);
const N = Number(nStr ?? 64);

const HEAD = 64 * 1024 * 1024;
const fd = openSync(file, 'r');
const head = Buffer.alloc(HEAD);
readSync(fd, head, 0, HEAD, 0);
const gguf = new GGUFParser(head.buffer.slice(head.byteOffset, head.byteOffset + HEAD)).parse();

const t = gguf.tensors.find((x) => x.name === tensorName);
if (!t) { console.error(`tensor ${tensorName} not found`); process.exit(1); }
console.error(`tensor ${t.name} type=${t.type} dims=${t.dims} offset=${t.offset} dataOffset=${gguf.dataOffset}`);

const nbytes = tensorByteSize(t);
const buf = Buffer.alloc(nbytes);
readSync(fd, buf, 0, nbytes, gguf.dataOffset + Number(t.offset));
const tp = new GGUFParser(buf.buffer.slice(buf.byteOffset, buf.byteOffset + nbytes));
const vals = tp.getTensorData({ ...t, offset: 0n }, 0);

console.log(Array.from(vals.slice(0, N)).map((v) => v.toPrecision(8)).join('\n'));
