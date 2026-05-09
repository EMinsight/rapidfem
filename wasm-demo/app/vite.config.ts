import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	// Vite dev server needs to serve the wasm-pack output (../pkg) as static assets
	// and apply correct MIME for .wasm. resolve.alias points "$wasm" at ../pkg.
	resolve: {
		alias: {
			$wasm: new URL('../pkg', import.meta.url).pathname
		}
	},
	server: {
		fs: {
			// allow importing files from outside this app dir (the sibling pkg/)
			allow: ['..', '../..', '../../..']
		}
	}
});
