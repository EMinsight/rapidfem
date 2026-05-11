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

function api_base(): string {
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
	positions: number[];  // flat xyz, 3 verts × 3 floats per tri
	normals: number[];
	material: string | null;
}

export interface GeometryPayload {
	kind: 'geometry';
	bbox: { min: [number, number, number]; max: [number, number, number] };
	entities: GeometryEntity[];
	stats: { n_entities: number; n_triangles: number; maxh: number };
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
		/** Per-(freq, port) flat [A0,B0,C0, A1,B1,C1, …] node phasor terms. */
		fields?: (number[] | null)[][];
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

export { viz_load_mesh, viz_sample } from './viz';
