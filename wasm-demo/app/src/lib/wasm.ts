/**
 * WASM solver wrapper. Loads ../pkg/rapidfem_wasm.js with cache-bust and
 * exposes a streaming sweep that emits one frequency at a time so the UI
 * can render progress + plot points as they arrive.
 *
 * Strategy: rather than refactor the Rust solver into a stateful streamer,
 * we rewrite the [frequency].values list per-call to a single point and
 * invoke run_sweep(mesh_bytes, single_freq_toml). Assembly is cheap (~30ms)
 * compared to the LU solve (~10s/freq) so re-doing it per frequency costs
 * negligible total time and keeps the WASM API unchanged.
 */

let cached: { init: (...args: any[]) => Promise<unknown>; run_sweep: any } | null = null;

async function loadWasm() {
	if (cached) return cached;
	const build = Date.now();
	// Vite serves ../pkg via fs.allow; use absolute fetch so the import
	// works from any route depth.
	const mod = await import(/* @vite-ignore */ `/pkg/rapidfem_wasm.js?v=${build}`);
	await mod.default({ module_or_path: `/pkg/rapidfem_wasm_bg.wasm?v=${build}` });
	cached = { init: mod.default, run_sweep: mod.run_sweep };
	return cached;
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
	const { run_sweep } = await loadWasm();
	const { mesh_bytes, config_toml, frequencies_hz, on_point, on_status, abort_signal } = cfg;
	for (let k = 0; k < frequencies_hz.length; k++) {
		if (abort_signal?.aborted) return;
		const f = frequencies_hz[k];
		on_status?.(`solving ${(f / 1e9).toFixed(1)} GHz (${k + 1}/${frequencies_hz.length})…`);
		// Yield to the browser so the UI can repaint between solves.
		await new Promise((r) => setTimeout(r, 0));
		const single_toml = override_frequency(config_toml, f);
		const t0 = performance.now();
		const result = run_sweep(mesh_bytes, single_toml) as {
			frequencies_hz: number[];
			n_driven: number;
			n_nodes: number;
			sparams_flat: number[];
			fields_flat: number[];
			solve_time_s: number;
		};
		const dt = (performance.now() - t0) / 1000;
		const n = result.n_driven;
		const nN = result.n_nodes;
		const stride = n * n * 2;
		const S: SMatrix = [];
		for (let obs = 0; obs < n; obs++) {
			const row: SParam[] = [];
			for (let exc = 0; exc < n; exc++) {
				row.push({
					re: result.sparams_flat[2 * (obs * n + exc)],
					im: result.sparams_flat[2 * (obs * n + exc) + 1]
				});
			}
			S.push(row);
		}
		// Field flat is laid out [exc][node] for this single-frequency call.
		const fields: Float32Array[] = [];
		for (let exc = 0; exc < n; exc++) {
			fields.push(
				new Float32Array(result.fields_flat.slice(exc * nN, (exc + 1) * nN))
			);
		}
		await on_point(k, frequencies_hz.length, { freq_hz: f, S, fields, solve_time_s: dt });
	}
	on_status?.('done');
}

export async function preload_wasm() {
	await loadWasm();
}
