import { defineConfig } from 'vite';
import { heroBuildAssetsPlugin } from './build_hero_assets.mjs';

export default defineConfig({
  base: './',
  plugins: [heroBuildAssetsPlugin()],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: { format: 'iife', rollupOptions: {output: {inlineDynamicImports: true}} },
  build: { target: 'es2022', assetsInlineLimit: 0 },
});
