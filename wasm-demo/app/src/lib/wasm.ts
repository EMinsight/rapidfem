/**
 * WASM solver wrapper. The actual solver runs in a Web Worker so the main
 * thread (UI / 3D viewer) stays responsive while the LU factorization grinds.
 *
 * Strategy: rewrite the [frequency].values list per-call to a single point
 * and dispatch one solve message per frequency to the worker. Assembly is
 * cheap (~30ms) compared to the LU solve (~10s/freq) so re-doing it per
 * frequency in the worker costs negligible total time.
 */

import SolverWorker from './solver.worker?worker';

let worker: Worker | null = null;
let next_msg_id = 1;
const pending = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void }>();
let init_promise: Promise<void> | null = null;

function ensure_worker(): Promise<void> {
	if (init_promise) return init_promise;
	init_promise = new Promise((resolve, reject) => {
		worker = new SolverWorker();
		worker.onmessage = (e: MessageEvent) => {
			const m = e.data;
			if (m.type === 'ready') resolve();
			else if (m.type === 'point') pending.get(m.id)?.resolve(m);
			else if (m.type === 'panic_log') console.error('[wasm panic]', m.message);
			else if (m.type === 'error') {
				if (m.id != null) pending.get(m.id)?.reject(new Error(m.message));
				else reject(new Error(m.message));
			}
		};
		worker.onerror = (e) => reject(new Error(e.message));
		worker.postMessage({ type: 'init', wasm_url: '/pkg' });
	});
	return init_promise;
}

function solve_one(mesh_bytes: Uint8Array, config_toml: string, freq_hz: number) {
	if (!worker) throw new Error('worker not initialized');
	const id = next_msg_id++;
	return new Promise<any>((resolve, reject) => {
		pending.set(id, { resolve, reject });
		// Slice mesh_bytes into a fresh ArrayBuffer we can transfer (zero-copy).
		const buf = new Uint8Array(mesh_bytes.byteLength);
		buf.set(mesh_bytes);
		worker!.postMessage(
			{ type: 'solve', id, mesh_bytes: buf.buffer, config_toml, freq_hz },
			[buf.buffer]
		);
	}).finally(() => pending.delete(id));
}

export async function preload_wasm() {
	await ensure_worker();
}

export function abort_solver() {
	if (worker) {
		worker.terminate();
		worker = null;
		init_promise = null;
		for (const p of pending.values()) p.reject(new Error('aborted'));
		pending.clear();
	}
}

export type SParam = { re: number; im: number };
export type SMatrix = SParam[][]; // [obs][exc]

export interface FrequencyResult {
	freq_hz: number;
	S: SMatrix;
	/** Per-excitation nodal |E| magnitudes (V/m).
	 *  fields[exc] has length = mesh.n_nodes. */
	fields: Float32Array[];
	solve_time_s: number;
}

export interface SweepConfig {
	mesh_bytes: Uint8Array;
	config_toml: string;     // contains [frequency] values = [...]
	frequencies_hz: number[]; // override the TOML's freq list
	on_point: (k: number, total: number, point: FrequencyResult) => void | Promise<void>;
	on_status?: (msg: string) => void;
	abort_signal?: AbortSignal;
}

/**
 * Replace the [frequency] section of a TOML config with a single-value list.
 * Naive line-based rewrite; matches the layout produced by builder.dump().
 */
function override_frequency(toml: string, freq_hz: number): string {
	const lines = toml.split('\n');
	const out: string[] = [];
	let in_freq = false;
	let wrote = false;
	for (const line of lines) {
		const stripped = line.trim();
		if (stripped.startsWith('[')) {
			if (in_freq && !wrote) {
				out.push(`values = [${freq_hz}]`);
				wrote = true;
			}
			in_freq = stripped === '[frequency]';
			out.push(line);
		} else if (in_freq && (stripped.startsWith('values') || stripped.startsWith('range'))) {
			if (!wrote) {
				out.push(`values = [${freq_hz}]`);
				wrote = true;
			}
			// drop the original line
		} else {
			out.push(line);
		}
	}
	if (in_freq && !wrote) out.push(`values = [${freq_hz}]`);
	return out.join('\n');
}

export async function run_streaming_sweep(cfg: SweepConfig) {
	await ensure_worker();
	const { mesh_bytes, config_toml, frequencies_hz, on_point, on_status, abort_signal } = cfg;
	for (let k = 0; k < frequencies_hz.length; k++) {
		if (abort_signal?.aborted) return;
		const f = frequencies_hz[k];
		on_status?.(`solving ${(f / 1e9).toFixed(1)} GHz (${k + 1}/${frequencies_hz.length})…`);
		const single_toml = override_frequency(config_toml, f);
		const result = await solve_one(mesh_bytes, single_toml, f);
		if (abort_signal?.aborted) return;
		const n = result.n_driven;
		const nN = result.n_nodes;
		const sp = result.sparams_flat as Float64Array;
		const ff = result.fields_flat as Float32Array;
		const S: SMatrix = [];
		for (let obs = 0; obs < n; obs++) {
			const row: SParam[] = [];
			for (let exc = 0; exc < n; exc++) {
				row.push({ re: sp[2 * (obs * n + exc)], im: sp[2 * (obs * n + exc) + 1] });
			}
			S.push(row);
		}
		const fields: Float32Array[] = [];
		for (let exc = 0; exc < n; exc++) {
			fields.push(ff.slice(exc * nN, (exc + 1) * nN));
		}
		await on_point(k, frequencies_hz.length, {
			freq_hz: f, S, fields, solve_time_s: result.solve_time_s
		});
	}
	on_status?.('done');
}
