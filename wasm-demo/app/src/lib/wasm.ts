/**
 * WASM solver wrapper. The actual solver runs in a Web Worker so the main
 * thread (UI / 3D viewer) stays responsive while the LU factorization grinds.
 *
 * One pipeline: TS builds a `MeshSpec` → worker meshes + solves frequency-by-
 * frequency in WASM (rapidfem-mesher + rapidfem). Mesh data flows back as
 * `MeshData` for the in-browser viewer; per-frequency S-params + nodal |E|
 * stream back via `on_point` callbacks.
 */

import SolverWorker from './solver.worker?worker';
import type { MeshData } from './msh';

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
			else if (
				m.type === 'point' || m.type === 'mesh' ||
				m.type === 'viz_mesh_ready' || m.type === 'viz_samples'
			) pending.get(m.id)?.resolve(m);
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

function solve_one_spec(spec_json: string, options_json: string, freq_hz: number) {
	if (!worker) throw new Error('worker not initialized');
	const id = next_msg_id++;
	return new Promise<any>((resolve, reject) => {
		pending.set(id, { resolve, reject });
		worker!.postMessage({ type: 'solve_spec', id, spec_json, options_json, freq_hz });
	}).finally(() => pending.delete(id));
}

export async function preload_wasm() {
	await ensure_worker();
}

/** Run only the mesher (no solve) and return a `MeshData` ready for the
 *  in-browser viewer. */
export async function mesh_spec(spec: object): Promise<MeshData> {
	await ensure_worker();
	const id = next_msg_id++;
	const r: any = await new Promise((resolve, reject) => {
		pending.set(id, { resolve, reject });
		worker!.postMessage({ type: 'mesh_spec', id, spec_json: JSON.stringify(spec) });
	}).finally(() => pending.delete(id));
	return convert_mesh_data(r.mesh);
}

/** Upload the mesh's nodes/tets into the worker's viz cache so subsequent
 *  point-cloud samples skip the CDF build. Call once per mesh load. */
export async function viz_load_mesh(mesh: MeshData): Promise<void> {
	await ensure_worker();
	const id = next_msg_id++;
	// Clone the arrays before transferring so the viewer's local copy stays
	// usable (transferable buffers get neutered on the sender side).
	const nodes = new Float64Array(mesh.nodes);
	const tets = new Uint32Array(mesh.tets);
	await new Promise<any>((resolve, reject) => {
		pending.set(id, { resolve, reject });
		worker!.postMessage(
			{ type: 'viz_set_mesh', id, nodes, tets },
			[nodes.buffer, tets.buffer]
		);
	}).finally(() => pending.delete(id));
}

/** Sample N volume points from the worker. Returns world-space positions and
 *  per-point phasor terms (A, B, C) — the GPU shader composites those with
 *  a phase uniform every frame to animate the wave. */
export async function viz_sample(
	field_abc: Float32Array, n: number
): Promise<{
	positions: Float32Array;
	abc: Float32Array;
	log_floor: number;
	log_range: number;
	field_range: { min: number; max: number; decades: number };
}> {
	await ensure_worker();
	const id = next_msg_id++;
	const f = new Float32Array(field_abc);
	const r: any = await new Promise((resolve, reject) => {
		pending.set(id, { resolve, reject });
		worker!.postMessage(
			{ type: 'viz_sample', id, field: f, n },
			[f.buffer]
		);
	}).finally(() => pending.delete(id));
	return {
		positions: r.positions as Float32Array,
		abc: r.abc as Float32Array,
		log_floor: r.log_floor,
		log_range: r.log_range,
		field_range: r.field_range,
	};
}

function convert_mesh_data(m: any): MeshData {
	const nodes = new Float64Array(m.nodes);
	const tets = new Uint32Array(m.tets);
	const tris = new Uint32Array(m.tris);
	const tri_phys = new Int32Array(m.tri_tag);
	const tet_phys = new Int32Array(m.tet_tag);
	const phys_names = new Map<number, string>();
	for (const [t, n] of m.tag_names as [number, string][]) phys_names.set(t, n);
	const phys_dim = new Map<number, number>();
	for (const [t, d] of m.tag_dim as [number, number][]) phys_dim.set(t, d);
	let xmin = Infinity, ymin = Infinity, zmin = Infinity;
	let xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
	for (let i = 0; i < nodes.length; i += 3) {
		const x = nodes[i], y = nodes[i + 1], z = nodes[i + 2];
		if (x < xmin) xmin = x; if (x > xmax) xmax = x;
		if (y < ymin) ymin = y; if (y > ymax) ymax = y;
		if (z < zmin) zmin = z; if (z > zmax) zmax = z;
	}
	return {
		nodes, tris, tri_phys, tets, tet_phys,
		phys_names, phys_dim,
		bbox: { min: [xmin, ymin, zmin], max: [xmax, ymax, zmax] }
	};
}

export interface SpecSweepConfig {
	spec: object;             // MeshSpec
	frequencies_hz: number[];
	port_z0?: number;
	materials?: Record<string, { er: number; conductivity?: number; tand?: number }>;
	on_point: (k: number, total: number, point: FrequencyResult) => void | Promise<void>;
	on_status?: (msg: string) => void;
	abort_signal?: AbortSignal;
}

/** Frequency-by-frequency sweep. Mesher runs once per frequency call (cheap
 *  vs the LU solve) inside `solve_from_spec`; results stream back through
 *  `on_point` so the UI updates incrementally. */
export async function run_streaming_sweep_spec(cfg: SpecSweepConfig) {
	await ensure_worker();
	const { spec, frequencies_hz, on_point, on_status, abort_signal } = cfg;
	const spec_json = JSON.stringify(spec);
	for (let k = 0; k < frequencies_hz.length; k++) {
		if (abort_signal?.aborted) return;
		const f = frequencies_hz[k];
		on_status?.(`meshing + solving ${(f / 1e9).toFixed(1)} GHz (${k + 1}/${frequencies_hz.length})…`);
		const opts = {
			frequencies_hz: [f],
			port_z0: cfg.port_z0 ?? 50,
			materials: cfg.materials ?? {}
		};
		const result = await solve_one_spec(spec_json, JSON.stringify(opts), f);
		if (abort_signal?.aborted) return;
		const n = result.n_driven;
		const nN = result.n_nodes;
		const sp = result.sparams_flat as Float64Array;
		const ff = result.fields_abc_flat as Float32Array;
		const S: SMatrix = [];
		for (let obs = 0; obs < n; obs++) {
			const row: SParam[] = [];
			for (let exc = 0; exc < n; exc++) {
				row.push({ re: sp[2 * (obs * n + exc)], im: sp[2 * (obs * n + exc) + 1] });
			}
			S.push(row);
		}
		// 3 floats per node (A, B, C) per excitation.
		const stride = nN * 3;
		const fields: Float32Array[] = [];
		for (let exc = 0; exc < n; exc++) fields.push(ff.slice(exc * stride, (exc + 1) * stride));
		await on_point(k, frequencies_hz.length, {
			freq_hz: f, S, fields, solve_time_s: result.solve_time_s
		});
	}
	on_status?.('done');
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
	/** Per-excitation nodal phasor terms (A, B, C). Each array is
	 *  `n_nodes × 3` floats, packed `[A0,B0,C0, A1,B1,C1, ...]`. */
	fields: Float32Array[];
	solve_time_s: number;
}
