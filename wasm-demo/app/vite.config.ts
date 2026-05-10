import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		sveltekit(),
		{
			name: 'no-cache-static-assets',
			configureServer(server) {
				server.middlewares.use((req, res, next) => {
					if (req.url?.startsWith('/examples/') || req.url?.startsWith('/pkg/')) {
						res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate');
						res.setHeader('Pragma', 'no-cache');
						res.setHeader('Expires', '0');
					}
					next();
				});
			}
		}
	],
	resolve: {
		alias: {
			$wasm: new URL('../pkg', import.meta.url).pathname
		}
	},
	server: {
		fs: {
			allow: ['..', '../..', '../../..']
		}
	}
});
