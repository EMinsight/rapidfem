import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		port: 5173,
		proxy: {
			// Forward API + WS to the Flask backend started by `rapidfem serve`.
			'/api': 'http://127.0.0.1:5174',
			'/ws': {
				target: 'ws://127.0.0.1:5174',
				ws: true,
			},
		},
	},
});
