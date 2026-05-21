/**
 * Static-demo mode flag.
 *
 * When the frontend is built with `VITE_STATIC_MODE=1` (set by the
 * GH-Pages deploy workflow), the app loads pre-baked example outputs
 * from `static/demo/` instead of talking to a Flask backend. The kernel
 * client is swapped for a replay implementation, the file browser only
 * shows baked examples, and every UI affordance that would mutate
 * server state is disabled.
 *
 * Single-source-of-truth: every consumer imports `IS_STATIC_MODE`
 * from here. Do NOT read `import.meta.env.VITE_STATIC_MODE` directly
 * anywhere else.
 */

// Vite injects `import.meta.env` at build time and stringifies env values
// ("1"/"true"). Comparing the literal directly - no `.toString()` chain -
// keeps this a build-time constant, so every `{#if IS_STATIC_MODE}` block
// (the marketing landing page, the analytics beacon, ...) is dead-code-
// eliminated from the pip-installed UI build rather than merely hidden.
export const IS_STATIC_MODE: boolean =
	import.meta.env.VITE_STATIC_MODE === '1' ||
	import.meta.env.VITE_STATIC_MODE === 'true';

/** Base URL prefix for fetching baked artefacts. */
export const DEMO_BASE = `${import.meta.env.BASE_URL ?? '/'}demo/`.replace(/\/+/g, '/');
