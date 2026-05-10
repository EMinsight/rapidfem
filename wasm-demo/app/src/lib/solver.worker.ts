/**
 * Web worker that hosts the rapidfem WASM solver. Runs frequency-by-frequency
 * on a background thread so the main thread (UI / 3D viewer) stays responsive.
 *
 * Protocol (main → worker):
 *   { type: 'init', wasm_url: string }
 *   { type: 'solve', id: number, mesh_bytes: ArrayBuffer, config_toml: string }
 * Worker → main:
 *   { type: 'ready' }
 *   { type: 'point', id, freq_hz, n_driven, n_nodes, sparams_flat, fields_flat, solve_time_s }
 *   { type: 'error', id?: number, message: string }
 */

let init: ((arg?: any) => Promise<unknown>) | null = null;
let run_sweep: ((mesh: Uint8Array, toml: string) => any) | null = null;
let ready = false;

// Capture WASM panic messages: console_error_panic_hook in the worker's
// rust crate writes panics to console.error. We tee them to postMessage so
// the main thread can show the actual reason instead of just "unreachable".
const orig_error = console.error.bind(console);
console.error = (...args: any[]) => {
	orig_error(...args);
	try {
		(self as any).postMessage({
			type: 'panic_log',
			message: args.map((a) => (typeof a === 'string' ? a : String(a))).join(' ')
		});
	} catch {}
};

self.onmessage = async (e: MessageEvent) => {
	const msg = e.data;
	try {
		if (msg.type === 'init') {
			const mod = await import(/* @vite-ignore */ msg.wasm_url + '/rapidfem_wasm.js?v=' + Date.now());
			init = mod.default;
			run_sweep = mod.run_sweep;
			await init!({ module_or_path: msg.wasm_url + '/rapidfem_wasm_bg.wasm?v=' + Date.now() });
			ready = true;
			(self as any).postMessage({ type: 'ready' });
		} else if (msg.type === 'solve') {
			if (!ready || !run_sweep) {
				(self as any).postMessage({ type: 'error', id: msg.id, message: 'worker not ready' });
				return;
			}
			const mesh_bytes = new Uint8Array(msg.mesh_bytes as ArrayBuffer);
			const t0 = performance.now();
			const r = run_sweep(mesh_bytes, msg.config_toml);
			const dt = (performance.now() - t0) / 1000;
			// Re-package as transferable Float64/Float32 arrays for zero-copy back.
			const sparams_flat = new Float64Array(r.sparams_flat);
			const fields_flat = new Float32Array(r.fields_flat);
			(self as any).postMessage(
				{
					type: 'point',
					id: msg.id,
					freq_hz: msg.freq_hz,
					n_driven: r.n_driven,
					n_nodes: r.n_nodes,
					sparams_flat,
					fields_flat,
					solve_time_s: dt
				},
				[sparams_flat.buffer, fields_flat.buffer]
			);
		}
	} catch (err) {
		(self as any).postMessage({
			type: 'error',
			id: msg.id,
			message: String(err && (err as any).stack ? (err as any).stack : err)
		});
	}
};
