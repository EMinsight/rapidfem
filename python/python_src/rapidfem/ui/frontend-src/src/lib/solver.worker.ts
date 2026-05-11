/**
 * Web worker that hosts the rapidfem WASM kernel. Off-main-thread so the UI
 * (3D viewer, plots) stays responsive while the LU factorization grinds.
 *
 * Protocol (main → worker):
 *   { type: 'init', wasm_url: string }
 *   { type: 'mesh_spec', id, spec_json }
 *   { type: 'solve_spec', id, spec_json, options_json, freq_hz }
 * Worker → main:
 *   { type: 'ready' }
 *   { type: 'mesh', id, mesh }
 *   { type: 'point', id, freq_hz, n_driven, n_nodes, sparams_flat, fields_flat, solve_time_s }
 *   { type: 'panic_log', message }   // teed Rust panics
 *   { type: 'error', id?, message }
 */

let init: ((arg?: any) => Promise<unknown>) | null = null;
let solve_from_spec: ((spec_json: string, options_json: string) => any) | null = null;
let mesh_from_spec: ((spec_json: string) => any) | null = null;
let ready = false;

/** Cached mesh-side data for field point-cloud sampling. Filled on
 *  `viz_set_mesh`, reused across many `viz_sample` calls so the heavy
 *  per-tet volume + CDF build runs once per mesh. */
let viz_mesh: {
	nodes: Float64Array;
	tets: Uint32Array;
	cdf: Float64Array;        // cumulative tet volume
	total_vol: number;
} | null = null;

// Tee WASM panic messages (console_error_panic_hook → console.error) to the
// main thread so the user sees the real reason, not just an `unreachable` trap.
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
			solve_from_spec = mod.solve_from_spec;
			mesh_from_spec = mod.mesh_from_spec;
			await init!({ module_or_path: msg.wasm_url + '/rapidfem_wasm_bg.wasm?v=' + Date.now() });
			ready = true;
			(self as any).postMessage({ type: 'ready' });
		} else if (msg.type === 'mesh_spec') {
			if (!ready || !mesh_from_spec) {
				(self as any).postMessage({ type: 'error', id: msg.id, message: 'worker not ready' });
				return;
			}
			const r = mesh_from_spec(msg.spec_json);
			(self as any).postMessage({ type: 'mesh', id: msg.id, mesh: r });
		} else if (msg.type === 'solve_spec') {
			if (!ready || !solve_from_spec) {
				(self as any).postMessage({ type: 'error', id: msg.id, message: 'worker not ready' });
				return;
			}
			const t0 = performance.now();
			const r = solve_from_spec(msg.spec_json, msg.options_json);
			const dt = (performance.now() - t0) / 1000;
			const sparams_flat = new Float64Array(r.sparams_flat);
			const fields_abc_flat = new Float32Array(r.fields_abc_flat);
			(self as any).postMessage(
				{
					type: 'point',
					id: msg.id,
					freq_hz: msg.freq_hz,
					n_driven: r.n_driven,
					n_nodes: r.n_nodes,
					sparams_flat,
					fields_abc_flat,
					solve_time_s: dt
				},
				[sparams_flat.buffer, fields_abc_flat.buffer]
			);
		} else if (msg.type === 'viz_set_mesh') {
			// Cache mesh-side aggregates (tet volume + CDF) so subsequent
			// `viz_sample` calls only do the random sampling pass.
			const nodes = msg.nodes as Float64Array;
			const tets = msg.tets as Uint32Array;
			const n_tets = tets.length / 4;
			const cdf = new Float64Array(n_tets);
			let acc = 0;
			for (let t = 0; t < n_tets; t++) {
				const i0 = tets[t * 4] * 3;
				const i1 = tets[t * 4 + 1] * 3;
				const i2 = tets[t * 4 + 2] * 3;
				const i3 = tets[t * 4 + 3] * 3;
				const ax = nodes[i1] - nodes[i0],     ay = nodes[i1 + 1] - nodes[i0 + 1], az = nodes[i1 + 2] - nodes[i0 + 2];
				const bx = nodes[i2] - nodes[i0],     by = nodes[i2 + 1] - nodes[i0 + 1], bz = nodes[i2 + 2] - nodes[i0 + 2];
				const cx = nodes[i3] - nodes[i0],     cy = nodes[i3 + 1] - nodes[i0 + 1], cz = nodes[i3 + 2] - nodes[i0 + 2];
				const vol = Math.abs(
					ax * (by * cz - bz * cy) -
					ay * (bx * cz - bz * cx) +
					az * (bx * cy - by * cx)
				) / 6;
				acc += vol;
				cdf[t] = acc;
			}
			viz_mesh = { nodes, tets, cdf, total_vol: acc };
			(self as any).postMessage({ type: 'viz_mesh_ready', id: msg.id });
		} else if (msg.type === 'viz_sample') {
			if (!viz_mesh) {
				(self as any).postMessage({ type: 'error', id: msg.id, message: 'viz mesh not set' });
				return;
			}
			// Expect 3 floats per node: (A, B, C) phasor terms.
			const field_abc = msg.field as Float32Array;
			const total_pts: number = msg.n;
			const DECADES = 6;
			const { nodes, tets, cdf, total_vol } = viz_mesh;
			const n_tets = tets.length / 4;

			// Per-tet mean (A, B, C) plus the global static-magnitude max
			// (max √(A+B)) for the log colorbar range.
			const tet_a = new Float32Array(n_tets);
			const tet_b = new Float32Array(n_tets);
			const tet_c = new Float32Array(n_tets);
			let max_v = 0;
			for (let t = 0; t < n_tets; t++) {
				const i0 = tets[t * 4] * 3;
				const i1 = tets[t * 4 + 1] * 3;
				const i2 = tets[t * 4 + 2] * 3;
				const i3 = tets[t * 4 + 3] * 3;
				const a = (field_abc[i0]     + field_abc[i1]     + field_abc[i2]     + field_abc[i3])     * 0.25;
				const b = (field_abc[i0 + 1] + field_abc[i1 + 1] + field_abc[i2 + 1] + field_abc[i3 + 1]) * 0.25;
				const c = (field_abc[i0 + 2] + field_abc[i1 + 2] + field_abc[i2 + 2] + field_abc[i3 + 2]) * 0.25;
				tet_a[t] = a;
				tet_b[t] = b;
				tet_c[t] = c;
				const m = Math.sqrt(a + b);
				if (m > max_v) max_v = m;
			}
			const log_max = Math.log10(max_v + 1e-30);
			const log_floor = log_max - DECADES;

			const positions = new Float32Array(total_pts * 3);
			const abc = new Float32Array(total_pts * 3);
			for (let p = 0; p < total_pts; p++) {
				const u = Math.random() * total_vol;
				let lo = 0, hi = n_tets - 1;
				while (lo < hi) {
					const mid = (lo + hi) >> 1;
					if (cdf[mid] < u) lo = mid + 1; else hi = mid;
				}
				const t = lo;
				const n0 = tets[t * 4] * 3, n1 = tets[t * 4 + 1] * 3,
				      n2 = tets[t * 4 + 2] * 3, n3 = tets[t * 4 + 3] * 3;
				let r1 = Math.random(), r2 = Math.random(), r3 = Math.random();
				if (r1 + r2 > 1) { r1 = 1 - r1; r2 = 1 - r2; }
				if (r2 + r3 > 1) { const t1 = r3; r3 = 1 - r1 - r2; r2 = 1 - t1; }
				else if (r1 + r2 + r3 > 1) { const t1 = r3; r3 = r1 + r2 + r3 - 1; r1 = 1 - r2 - t1; }
				const r0 = 1 - r1 - r2 - r3;
				positions[p * 3 + 0] = r0 * nodes[n0]     + r1 * nodes[n1]     + r2 * nodes[n2]     + r3 * nodes[n3];
				positions[p * 3 + 1] = r0 * nodes[n0 + 1] + r1 * nodes[n1 + 1] + r2 * nodes[n2 + 1] + r3 * nodes[n3 + 1];
				positions[p * 3 + 2] = r0 * nodes[n0 + 2] + r1 * nodes[n1 + 2] + r2 * nodes[n2 + 2] + r3 * nodes[n3 + 2];
				abc[p * 3 + 0] = tet_a[t];
				abc[p * 3 + 1] = tet_b[t];
				abc[p * 3 + 2] = tet_c[t];
			}

			(self as any).postMessage(
				{
					type: 'viz_samples',
					id: msg.id,
					positions,
					abc,
					log_floor,
					log_range: log_max - log_floor,
					field_range: { min: Math.pow(10, log_floor), max: max_v, decades: DECADES }
				},
				[positions.buffer, abc.buffer]
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
