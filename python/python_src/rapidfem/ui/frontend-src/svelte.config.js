import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	kit: {
		adapter: adapter({
			// Write the build directly into the python package's frontend/dist/
			// so `pip install -e` (and the in-CI maturin step) can ship it
			// via importlib.resources without an extra copy step.
			pages: '../frontend/dist',
			assets: '../frontend/dist',
			fallback: 'index.html',
		}),
		// SPA-style: every route resolves to fallback at runtime, no server.
		prerender: { entries: [] },
	},
	vitePlugin: {
		dynamicCompileOptions: ({ filename }) =>
			filename.includes('node_modules') ? undefined : { runes: true }
	}
};

export default config;
