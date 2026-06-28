import { defineConfig } from 'vite';

// Library build: bundle the engine (src/index.ts) into a single self-contained
// ESM file with all WGSL (?raw imports) inlined. Output is committed into the
// parent repo's browserlab playground (custom-kernels/ is gitignored there, so
// the deployed static site needs a pre-built bundle, not the source).
export default defineConfig({
  build: {
    lib: {
      entry: 'src/index.ts',
      formats: ['es'],
      fileName: () => 'browserlab-engine.js',
    },
    outDir: 'dist-lib',
    emptyOutDir: true,
    target: 'es2022',
    minify: true,
  },
});
