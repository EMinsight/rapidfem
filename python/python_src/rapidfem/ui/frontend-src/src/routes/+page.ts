// Pre-render the landing page so GH-Pages serves it as a real
// `index.html` at /. SSR stays off because the page uses the
// <fem-viewer> custom element, which only exists at runtime.
export const prerender = true;
export const ssr = false;
