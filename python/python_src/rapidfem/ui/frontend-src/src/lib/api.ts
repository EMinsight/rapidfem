/**
 * Backend client for the rapidfem UI.
 *
 * Talks to the Flask app started by `rapidfem serve`:
 *   POST /api/run    — exec user code, return rendered Geometry payloads
 *   POST /api/mesh   — gmsh the captured Geometry, return tet/tri mesh
 *   POST /api/solve  — build + run_sweep, return frequencies + sparams
 *   GET  /api/files  — list .py files in the workdir
 *   GET  /api/files/<path>, PUT /api/files/<path>
 *   WS   /ws         — pub/sub event stream (stage_start, stage_end, …)
 */

import type { MeshData } from './msh';

// ── URL base ──────────────────────────────────────────────────────────────

// In production the SvelteKit static build is served by the same Flask
// process at /, so relative URLs work. In dev (vite at :5173) we point at
// the Flask port directly.
const DEV_BACKEND = 'http://127.0.0.1:5174';

export function api_base(): string {
	if (typeof window !== 'undefined' && window.location.port === '5173') {
		return DEV_BACKEND;
	}
	return '';
}

function ws_url(): string {
	if (typeof window === 'undefined') return '';
	const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
	if (window.location.port === '5173') return `${proto}//127.0.0.1:5174/ws`;
	return `${proto}//${window.location.host}/ws`;
}

// ── Shared types ──────────────────────────────────────────────────────────

export type SParam = { re: number; im: number };
export type SMatrix = SParam[][]; // [obs][exc]

export interface PythonError {
	type: string;
	message: string;
	traceback: string;
}

export interface RunResponse {
	ok: boolean;
	error?: PythonError;
	stdout: string;
	stderr: string;
	captures: Array<{
		name: string;
		kind: 'geometry' | 'builder' | 'simulation' | 'result' | 'unknown';
		payload?: GeometryPayload;
		error?: PythonError;
	}>;
}

export interface GeometryEntity {
	name: string;
	tag: number;
	dim: number;
	color: [number, number, number];
	// Triangle mode (after g.mesh()): flat xyz, 3 verts × 3 floats per tri.
	positions?: number[];
	normals?: number[];
	// Wireframe mode (before g.mesh()): flat xyz pairs, 2 verts per segment.
	lines?: number[];
	material: string | null;
}

export interface GeometryPayload {
	kind: 'geometry';
	wireframe?: boolean;
	bbox: { min: [number, number, number]; max: [number, number, number] };
	entities: GeometryEntity[];
	stats: { n_entities: number; n_segments?: number; n_triangles?: number; maxh: number };
}

export interface MeshPayload {
	kind: 'mesh';
	bbox: { min: [number, number, number]; max: [number, number, number] };
	nodes: number[];           // flat xyz
	tris: number[];            // flat 3 × idx
	tri_phys: number[];
	tets: number[];
	tet_phys: number[];
	phys_names: Record<string, string>;
	phys_dim: Record<string, number>;
	name_to_tag: Record<string, number>;
	stats: { n_nodes: number; n_tets: number; n_tris: number; mesh_time_s: number; msh_bytes: number };
}

export interface MeshResponse {
	ok: boolean;
	error?: PythonError;
	stdout: string;
	stderr: string;
	mesh?: MeshPayload;
	name?: string;
}

export interface SolveResponse {
	ok: boolean;
	error?: PythonError;
	stdout: string;
	stderr: string;
	result?: {
		frequencies: number[];
		sparams: number[][][][];  // [n_freq][n_p][n_p][re,im]
		n_driven: number;
		n_freq: number;
		n_dofs: number;
		n_tets: number;
		solve_time_s: number;
		/** Per-(freq, port) flat [A0,B0,C0, A1,B1,C1, …] node phasor terms
		 *  for the E-field channel. */
		fields?: (number[] | null)[][];
		/** Same shape as `fields`, conduction current density J = σE. Zero
		 *  in PEC / lossless regions; null when not computed. */
		fields_j?: (number[] | null)[][] | null;
		/** Same shape as `fields`, magnetic field H = ∇×E / (jωμ). */
		fields_h?: (number[] | null)[][] | null;
	};
	mesh?: MeshPayload;
	name?: string;
}

export interface FileEntry {
	path: string;
	size: number;
	mtime: number;
}

// ── HTTP helpers ──────────────────────────────────────────────────────────

async function post_json<T>(path: string, body: unknown): Promise<T> {
	const res = await fetch(api_base() + path, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
	if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
	return res.json();
}

async function get_json<T>(path: string): Promise<T> {
	const res = await fetch(api_base() + path);
	if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
	return res.json();
}

// ── Public API ────────────────────────────────────────────────────────────

export async function runCode(code: string): Promise<RunResponse> {
	return post_json<RunResponse>('/api/run', { code });
}

export interface CellResponse {
	ok: boolean;
	error?: PythonError;
	stdout: string;
	stderr: string;
	captures: RunResponse['captures'];
	mesh?: MeshPayload | null;
	result?: {
		frequencies: number[];
		sparams: number[][][][];
		n_driven: number;
		n_freq: number;
		n_dofs: number;
		n_tets: number;
		solve_time_s: number;
		fields?: (number[] | null)[][];
		fields_j?: (number[] | null)[][] | null;
		fields_h?: (number[] | null)[][] | null;
	} | null;
}

export async function runCell(file: string, code: string, reset: boolean = false): Promise<CellResponse> {
	return post_json<CellResponse>('/api/cell/run', { file, code, reset });
}

export async function resetKernel(file: string): Promise<{ ok: boolean }> {
	return post_json<{ ok: boolean }>('/api/cell/reset', { file });
}

export async function listExamples(): Promise<{ examples: { name: string }[] }> {
	return get_json<{ examples: { name: string }[] }>('/api/examples');
}

export async function readExample(name: string): Promise<string> {
	// In static-demo mode the example source lives inside the baked JSON.
	const { IS_STATIC_MODE, DEMO_BASE } = await import('./static_mode');
	if (IS_STATIC_MODE) {
		const stem = name.replace(/\.py$/, '');
		const r = await fetch(`${DEMO_BASE}${stem}.json`);
		if (!r.ok) throw new Error(`baked example fetch failed: ${r.status}`);
		const baked = await r.json() as { source: string };
		return baked.source;
	}
	const r = await get_json<{ ok: boolean; content?: string; error?: string }>(`/api/examples/${encodeURIComponent(name)}`);
	if (!r.ok || r.content === undefined) throw new Error(r.error ?? 'example not found');
	return r.content;
}

export async function meshGeometry(code: string, opts: { maxh?: number; geometry_name?: string } = {}): Promise<MeshResponse> {
	return post_json<MeshResponse>('/api/mesh', { code, ...opts });
}

export async function solve(code: string, opts: { builder_name?: string } = {}): Promise<SolveResponse> {
	return post_json<SolveResponse>('/api/solve', { code, ...opts });
}

export async function listFiles(): Promise<{ workdir: string; files: FileEntry[] }> {
	return get_json<{ workdir: string; files: FileEntry[] }>('/api/files');
}

export async function readFile(path: string): Promise<string> {
	const r = await get_json<{ ok: boolean; content?: string; error?: string }>(
		`/api/files/${encodeURI(path)}`,
	);
	if (!r.ok || r.content === undefined) throw new Error(r.error ?? 'read failed');
	return r.content;
}

export async function writeFile(path: string, content: string): Promise<void> {
	const res = await fetch(api_base() + `/api/files/${encodeURI(path)}`, {
		method: 'PUT',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ content }),
	});
	if (!res.ok) throw new Error(`writeFile ${path}: HTTP ${res.status}`);
}

export async function deleteFile(path: string): Promise<void> {
	const res = await fetch(api_base() + `/api/files/${encodeURI(path)}`, { method: 'DELETE' });
	if (!res.ok) throw new Error(`deleteFile ${path}: HTTP ${res.status}`);
}

export async function renameFile(from: string, to: string): Promise<void> {
	return post_json('/api/files/rename', { from, to });
}

export async function health(): Promise<{ ok: boolean; workdir: string; frontend_bundled: boolean }> {
	return get_json('/api/health');
}

// ── WebSocket bus ─────────────────────────────────────────────────────────

export type BusEvent =
	| { kind: 'hello'; ok: boolean }
	| { kind: 'stage_start'; stage: string }
	| { kind: 'stage_end'; stage: string; ok: boolean; [k: string]: unknown }
	| { kind: 'log'; line: string }
	| { kind: string; [k: string]: unknown };

export function subscribeBus(handler: (e: BusEvent) => void): () => void {
	if (typeof window === 'undefined') return () => {};
	let stopped = false;
	let ws: WebSocket | null = null;
	let reconnect_timer: ReturnType<typeof setTimeout> | null = null;

	const connect = () => {
		if (stopped) return;
		try {
			ws = new WebSocket(ws_url());
		} catch (err) {
			console.warn('[bus] open failed', err);
			schedule_reconnect();
			return;
		}
		ws.onmessage = (m) => {
			try {
				handler(JSON.parse(m.data) as BusEvent);
			} catch (err) {
				console.warn('[bus] bad payload', err);
			}
		};
		ws.onerror = () => {};
		ws.onclose = () => {
			ws = null;
			schedule_reconnect();
		};
	};
	const schedule_reconnect = () => {
		if (stopped || reconnect_timer) return;
		reconnect_timer = setTimeout(() => {
			reconnect_timer = null;
			connect();
		}, 1500);
	};

	connect();
	return () => {
		stopped = true;
		if (reconnect_timer) clearTimeout(reconnect_timer);
		ws?.close();
	};
}

// ── Adapters: server payload → frontend types ─────────────────────────────

export function meshPayloadToMeshData(p: MeshPayload): MeshData {
	const nodes = new Float64Array(p.nodes);
	const tris = new Uint32Array(p.tris);
	const tri_phys = new Int32Array(p.tri_phys);
	const tets = new Uint32Array(p.tets);
	const tet_phys = new Int32Array(p.tet_phys);
	const phys_names = new Map<number, string>();
	for (const [k, v] of Object.entries(p.phys_names)) phys_names.set(Number(k), v);
	const phys_dim = new Map<number, number>();
	for (const [k, v] of Object.entries(p.phys_dim)) phys_dim.set(Number(k), v);
	return {
		nodes, tris, tri_phys, tets, tet_phys,
		phys_names, phys_dim,
		bbox: { min: [...p.bbox.min] as [number, number, number], max: [...p.bbox.max] as [number, number, number] },
	};
}

export function sparamsToSMatrices(s: number[][][][]): SMatrix[] {
	return s.map((freq_mat) =>
		freq_mat.map((row) => row.map(([re, im]) => ({ re, im }))),
	);
}

// ── Time-domain display payloads ──────────────────────────────────────────
// Emitted by rapidfem.show() for the ProblemTD verb results — see
// api._td_*_payload on the Python side.

/** One trace of a time-series plot. A time-domain probe carries a real
 *  `y`; a frequency-domain transfer function carries a complex
 *  `y_re` / `y_im` pair. */
export interface TdSeries {
	label: string;
	y?: number[];
	y_re?: number[];
	y_im?: number[];
}

/** `td_timeseries` payload — driven_transient probe signals (`domain:'time'`)
 *  or a transfer function (`domain:'freq'`). */
export interface TdTimeSeriesPayload {
	domain: 'time' | 'freq';
	x_label: string;
	x: number[];
	series: TdSeries[];
	source_label: string;
}

/** `td_result` payload — a time-domain modal-port scattering matrix. Carries
 *  the same nested-list shape as the frequency-domain result, so it feeds the
 *  existing S-parameter panel unchanged. */
export interface TdResultPayload {
	frequencies: number[];
	sparams: number[][][][];
	n_port: number;
	n_freq: number;
}

/** `td_trajectory` payload — a field-animation point cloud. One point per
 *  sampled DG node; `frames_e` / `frames_h` carry a per-point |E| / |H|
 *  magnitude per snapshot. `field_max` is the global per-channel maximum,
 *  for a colour scale held fixed across the whole animation. */
export interface TdTrajectoryPayload {
	points: number[];          // flat xyz
	n_points: number;
	n_elem: number;
	bbox: { min: [number, number, number]; max: [number, number, number] };
	n_snapshots: number;
	times: number[];
	field_max: { E: number; H: number };
	frames_e: number[][];
	frames_h: number[][];
}

export { viz_load_mesh, viz_sample } from './viz';
