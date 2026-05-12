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

// Vite injects `import.meta.env` at build time. The flag is a string
// ("1"/"true") because Vite stringifies all env values; coerce to bool.
const raw = (import.meta.env.VITE_STATIC_MODE ?? '').toString().toLowerCase();
export const IS_STATIC_MODE: boolean = raw === '1' || raw === 'true';

/** Base URL prefix for fetching baked artefacts. */
export const DEMO_BASE = `${import.meta.env.BASE_URL ?? '/'}demo/`.replace(/\/+/g, '/');
