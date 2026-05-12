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
			// `404.html` is what GH-Pages serves for any unknown path —
			// using it as the SPA fallback lets deep links like /demo work
			// without prerendering every route. The root index.html is
			// produced from the prerendered '/' route below.
			fallback: '404.html',
		}),
		prerender: { entries: ['/'] },
	},
	vitePlugin: {
		dynamicCompileOptions: ({ filename }) =>
			filename.includes('node_modules') ? undefined : { runes: true }
	}
};

export default config;
