import { defineConfig } from 'vite';
import { resolve } from 'path';

// Build configuration for the standalone <fem-viewer> web component
// bundle. The SvelteKit build (vite.config.ts) emits the notebook UI;
// this one emits a small library you can `<script src=...>` from any
// HTML page to drop an embedded rapidfem viewer in.
//
// Run with: `npm run build:embed`
// Output:   static/embed/fem-viewer.js (lands in dist/embed/ after the
//           main build via SvelteKit's static-adapter copy).

export default defineConfig({
	build: {
		lib: {
			entry: resolve(__dirname, 'src/embed/fem-viewer.ts'),
			name: 'FemViewer',
			fileName: 'fem-viewer',
			formats: ['iife'],
		},
		outDir: 'static/embed',
		emptyOutDir: false,
		rollupOptions: {
			output: {
				entryFileNames: 'fem-viewer.js',
			},
		},
	},
	resolve: {
		alias: {
			'$lib': resolve(__dirname, 'src/lib'),
		},
	},
});
