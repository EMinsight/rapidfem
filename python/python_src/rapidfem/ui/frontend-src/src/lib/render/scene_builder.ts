/**
 * Shared scene builder — translates a mesh payload into GL state via
 * `canvas3d` primitives. Used by both `MeshViewer.svelte` (the in-app
 * viewer) and `embed/fem-viewer.ts` (the standalone web-component
 * embed) so the two produce bit-identical renderings.
 *
 * Pipeline (matches `MeshViewer.rebuild()` exactly):
 *
 *   1. Named surface tris (PEC walls, ports, ground, etc.):
 *      group by tri_phys, classify by name, color by kind+name
 *   2. Implicit volume hulls (substrate, air, PML shells):
 *      per-volume outer faces from tet_phys, dielectric color,
 *      polygon-offset to push behind coplanar conductor surfaces
 *   3. Optional wireframe: every named-surface tri's edges,
 *      one dim color
 *   4. Optional field point cloud: caller-supplied
 *
 * No Svelte deps — pure TS. Safe to call from a web component or a
 * Svelte rebuild $effect alike.
 */

import {
	addMesh, addLineMesh, setBBox, setSplatCloud,
	type GLState,
} from './canvas3d';
import { buildVolumeBoundaries } from './mesh_scene';

// ── Mesh-payload contract ────────────────────────────────────────────

export interface SceneMesh {
	nodes: number[] | Float32Array | Float64Array;
	tris: number[];
	tri_phys: number[];
	tets: number[];
	tet_phys: number[];
	phys_names: Map<number, string> | Record<string, string>;
	phys_dim?: Map<number, number> | Record<string, number>;
	bbox: { min: [number, number, number]; max: [number, number, number] };
}

// Allow `phys_names` from JSON (object keyed by string) or from the
// in-app code path (Map keyed by number).
function physName(m: SceneMesh, tag: number): string {
	if (m.phys_names instanceof Map) return m.phys_names.get(tag) ?? '';
	return (m.phys_names as Record<string, string>)[String(tag)] ?? '';
}

function physDim(m: SceneMesh, tag: number): number {
	if (!m.phys_dim) return 2;
	if (m.phys_dim instanceof Map) return m.phys_dim.get(tag) ?? 2;
	const v = (m.phys_dim as Record<string, number>)[String(tag)];
	return v ?? 2;
}

// ── Material classification + coloring ───────────────────────────────

export type Kind = 'dielectric' | 'conductor' | 'port' | 'gnd';

export function classify(name: string): Kind | null {
	if (name === 'abc' || name.startsWith('_mat_')) return null;
	if (name === 'substrate' || name === 'oxide' || name === 'air') return 'dielectric';
	if (name.endsWith('_gnd') || name === 'gnd' || name === 'ground') return 'gnd';
	if (
		name === 'p1' || name === 'p2' || /^p\d+$/.test(name) ||
		name.startsWith('port') || name.endsWith('_port')
	) return 'port';
	return 'conductor';
}

function hex(s: string): [number, number, number] {
	return [
		parseInt(s.slice(1, 3), 16) / 255,
		parseInt(s.slice(3, 5), 16) / 255,
		parseInt(s.slice(5, 7), 16) / 255,
	];
}

const FIXED_CONDUCTOR_COLORS: Record<string, string> = {
	met5: '#e8944a', met4: '#f0b86a', met3: '#c4c46b',
	met2: '#9bc28b', met1: '#7b9fb8', li1:  '#5a8caa',
	via5: '#d9513c', via4: '#e5634f', via3: '#bf4233',
	via2: '#9d3526', via1: '#7c281b', mcon: '#aa6b40',
};

export function colorFor(kind: Kind, name: string): [number, number, number] {
	if (kind === 'dielectric') {
		if (name === 'substrate') return hex('#4a9ec2');
		if (name === 'oxide') return hex('#7b5e8a');
		return hex('#5a6470');
	}
	if (kind === 'gnd') return hex('#5aad78');
	if (kind === 'port') return hex('#d9513c');           // accent
	return hex(FIXED_CONDUCTOR_COLORS[name] ?? '#e8944a'); // accent-secondary default
}

// ── Push one tri-group with high-precision normals ───────────────────

function pushGroup(
	state: GLState,
	mesh: SceneMesh,
	idx: number[],
	color: [number, number, number],
	tag: number,
	depthOffset: [number, number] | undefined,
	fieldNorm: Float32Array | null,
): void {
	const ntri = idx.length / 3;
	const pos64 = new Float64Array(ntri * 9);
	for (let t = 0; t < ntri; t++) {
		for (let v = 0; v < 3; v++) {
			const ni = idx[t * 3 + v] * 3;
			pos64[t * 9 + v * 3 + 0] = mesh.nodes[ni];
			pos64[t * 9 + v * 3 + 1] = mesh.nodes[ni + 1];
			pos64[t * 9 + v * 3 + 2] = mesh.nodes[ni + 2];
		}
	}
	const norm64 = new Float64Array(ntri * 9);
	for (let t = 0; t < ntri; t++) {
		const i = t * 9;
		const ax = pos64[i + 0], ay = pos64[i + 1], az = pos64[i + 2];
		const bx = pos64[i + 3], by = pos64[i + 4], bz = pos64[i + 5];
		const cx = pos64[i + 6], cy = pos64[i + 7], cz = pos64[i + 8];
		const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
		const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
		let nx = e1y * e2z - e1z * e2y;
		let ny = e1z * e2x - e1x * e2z;
		let nz = e1x * e2y - e1y * e2x;
		const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
		nx /= l; ny /= l; nz /= l;
		// Snap axis-aligned normals so coplanar walls share an exact value.
		if (Math.abs(nx) > 0.9999)      { nx = Math.sign(nx); ny = 0; nz = 0; }
		else if (Math.abs(ny) > 0.9999) { ny = Math.sign(ny); nx = 0; nz = 0; }
		else if (Math.abs(nz) > 0.9999) { nz = Math.sign(nz); nx = 0; ny = 0; }
		for (let k = 0; k < 3; k++) {
			norm64[i + k * 3 + 0] = nx;
			norm64[i + k * 3 + 1] = ny;
			norm64[i + k * 3 + 2] = nz;
		}
	}
	let scalars: Float32Array | undefined;
	if (fieldNorm) {
		scalars = new Float32Array(ntri * 3);
		for (let t = 0; t < ntri; t++) {
			for (let v = 0; v < 3; v++) {
				scalars[t * 3 + v] = fieldNorm[idx[t * 3 + v]];
			}
		}
	}
	addMesh(
		state,
		Float32Array.from(pos64),
		Float32Array.from(norm64),
		color, tag, depthOffset, scalars,
	);
}

// ── Wireframe edges of every named surface tri ───────────────────────

function buildWireEdges(mesh: SceneMesh): Float32Array {
	const seen = new Set<bigint>();
	const out: number[] = [];
	const push = (a: number, b: number) => {
		const lo = a < b ? a : b, hi = a < b ? b : a;
		const k = (BigInt(lo) << 32n) | BigInt(hi);
		if (seen.has(k)) return;
		seen.add(k);
		out.push(
			mesh.nodes[a * 3], mesh.nodes[a * 3 + 1], mesh.nodes[a * 3 + 2],
			mesh.nodes[b * 3], mesh.nodes[b * 3 + 1], mesh.nodes[b * 3 + 2],
		);
	};
	const n_tris = mesh.tris.length / 3;
	for (let t = 0; t < n_tris; t++) {
		const a = mesh.tris[t * 3], b = mesh.tris[t * 3 + 1], c = mesh.tris[t * 3 + 2];
		push(a, b); push(b, c); push(c, a);
	}
	return Float32Array.from(out);
}

// ── Public API ───────────────────────────────────────────────────────

export interface BuildSceneConfig {
	showFaces?: boolean;           // named surfaces + volume hulls (default true)
	showWire?: boolean;            // edge wireframe (default false)
	showField?: boolean;           // point cloud (caller sets it separately)
	/** Optional per-node normalised field magnitude for vertex-tinted faces.
	 *  Drives the inferno colormap on the mesh surfaces in field mode. */
	fieldNorm?: Float32Array | null;
}

/**
 * Wipe the GL state's previous scene contents and rebuild from `mesh`.
 *
 * NOTE: the caller is responsible for `clearMeshes()` BEFORE this — we
 * don't do it here so callers can compose multiple scenes (geometry +
 * wireframe overlay) if they want. We do call `setBBox` so the camera
 * fitter has fresh bounds.
 *
 * The field point cloud is not set here; the in-app viewer hands it
 * off to a worker and the embed builds a synchronous tet-centroid
 * sample — both call setSplatCloud themselves.
 */
export const WIRE_TAG = -1;

export function buildScene(
	state: GLState,
	mesh: SceneMesh,
	config: BuildSceneConfig = {},
): { faceTags: number[]; wireTag: number | null } {
	const showFaces = config.showFaces ?? true;
	const showWire = config.showWire ?? false;
	const fieldNorm = config.fieldNorm ?? null;
	const faceTags: number[] = [];
	let wireTag: number | null = null;

	setBBox(state, mesh.bbox.min, mesh.bbox.max);

	if (showFaces) {
		// 1) Named surface tris (conductors / ports / gnd / ABC are skipped).
		const bySurf = new Map<number, number[]>();
		const n_tris = mesh.tri_phys.length;
		for (let i = 0; i < n_tris; i++) {
			const tag = mesh.tri_phys[i];
			if (!tag || physDim(mesh, tag) !== 2) continue;
			let arr = bySurf.get(tag);
			if (!arr) { arr = []; bySurf.set(tag, arr); }
			arr.push(mesh.tris[i * 3], mesh.tris[i * 3 + 1], mesh.tris[i * 3 + 2]);
		}
		for (const [tag, idx] of bySurf.entries()) {
			const name = physName(mesh, tag);
			const kind = classify(name);
			if (!kind) continue;
			pushGroup(state, mesh, idx, colorFor(kind, name), tag, undefined, fieldNorm);
			faceTags.push(tag);
		}
		// 2) Implicit volume hulls — substrate / air / PML shells. Push
		//    behind via polygon offset so coplanar conductors win the
		//    depth test cleanly. (In field-mode the colormap renders all
		//    surfaces by |E| anyway so the offset doesn't matter.)
		const volBoundaries = buildVolumeBoundaries(mesh);
		for (const [vtag, idx] of volBoundaries.entries()) {
			const name = physName(mesh, vtag);
			if (!name || name.startsWith('_mat_')) continue;
			const offset: [number, number] | undefined = fieldNorm ? undefined : [2, 2];
			pushGroup(state, mesh, idx, colorFor('dielectric', name), vtag, offset, fieldNorm);
			faceTags.push(vtag);
		}
	}

	if (showWire) {
		const edges = buildWireEdges(mesh);
		// Dim grey — matches the line color MeshViewer uses for its mesh
		// wireframe overlay so embed + in-app look identical.
		addLineMesh(state, edges, hex('#3a3a44'), WIRE_TAG);
		wireTag = WIRE_TAG;
	}

	return { faceTags, wireTag };
}

/** Convenience: wipe the field splat cloud. Callers use this when toggling
 *  out of field mode. */
export function clearFieldCloud(state: GLState): void {
	setSplatCloud(state, new Float32Array(0), new Float32Array(0), new Float32Array(0));
}

// ── Volumetric field splat sampling ──────────────────────────────────
//
// Same algorithm as `$lib/viz.ts` (the sampler the in-app MeshViewer uses),
// but in-line and synchronous for the embed which doesn't carry a worker.
// Sampling is energy-weighted: per-tet draw probability is
// `volume × (floor + energy)`, so splats concentrate where the field is
// strong, and each sample carries the containing tet's characteristic size
// as its world-space Gaussian σ.

/** σ as a fraction of the containing tet's mean edge length — matches
 *  `viz.ts:SIGMA_FACTOR`. */
const SIGMA_FACTOR = 0.35;
/** Energy coverage floor — matches `viz.ts:ENERGY_FLOOR`. */
const ENERGY_FLOOR = 0.05;

interface TetSampleCache {
	vols: Float64Array;     // |tet volume|, per tet
	sizes: Float64Array;    // characteristic length (mean edge) × SIGMA_FACTOR
}

function buildTetCache(mesh: SceneMesh): TetSampleCache {
	const tets = mesh.tets;
	const nodes = mesh.nodes;
	const n = tets.length / 4;
	const vols = new Float64Array(n);
	const sizes = new Float64Array(n);
	for (let t = 0; t < n; t++) {
		const a = tets[t * 4], b = tets[t * 4 + 1],
		      c = tets[t * 4 + 2], d = tets[t * 4 + 3];
		const ax = nodes[a * 3], ay = nodes[a * 3 + 1], az = nodes[a * 3 + 2];
		const bx = nodes[b * 3], by = nodes[b * 3 + 1], bz = nodes[b * 3 + 2];
		const cx = nodes[c * 3], cy = nodes[c * 3 + 1], cz = nodes[c * 3 + 2];
		const dx = nodes[d * 3], dy = nodes[d * 3 + 1], dz = nodes[d * 3 + 2];
		const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
		const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
		const e3x = dx - ax, e3y = dy - ay, e3z = dz - az;
		const det = e1x * (e2y * e3z - e2z * e3y)
		          - e1y * (e2x * e3z - e2z * e3x)
		          + e1z * (e2x * e3y - e2y * e3x);
		vols[t] = Math.abs(det) / 6;
		// Mean of the 6 edge lengths.
		const px = [ax, bx, cx, dx], py = [ay, by, cy, dy], pz = [az, bz, cz, dz];
		let edgeSum = 0;
		for (let i = 0; i < 4; i++) {
			for (let j = i + 1; j < 4; j++) {
				const lx = px[i] - px[j], ly = py[i] - py[j], lz = pz[i] - pz[j];
				edgeSum += Math.sqrt(lx * lx + ly * ly + lz * lz);
			}
		}
		sizes[t] = (edgeSum / 6) * SIGMA_FACTOR;
	}
	return { vols, sizes };
}

function bsearchCdf(cdf: Float64Array, target: number): number {
	let lo = 0, hi = cdf.length - 1;
	while (lo < hi) {
		const mid = (lo + hi) >>> 1;
		if (cdf[mid] < target) lo = mid + 1;
		else hi = mid;
	}
	return lo;
}

/** Uniform barycentric weights for a tetrahedron (sorted-triple trick). */
function uniformBary(out: [number, number, number, number]): void {
	let s = Math.random();
	let t = Math.random();
	let u = Math.random();
	if (s > t) [s, t] = [t, s];
	if (t > u) [t, u] = [u, t];
	if (s > t) [s, t] = [t, s];
	out[0] = s;
	out[1] = t - s;
	out[2] = u - t;
	out[3] = 1 - u;
}

/** Energy-weighted random sampling of N field splats, with (A, B, C) phasor
 *  coefficients interpolated by barycentric weights and a per-splat σ from
 *  the containing tet. Same output shape `viz.ts:viz_sample` produces (plus
 *  maxE2/minE2 the embed uses for its colour range) — drop the positions /
 *  abc / sigma straight into `setSplatCloud`. */
export function sampleFieldCloud(
	mesh: SceneMesh,
	fieldAbc: number[] | Float32Array,
	n: number,
): { positions: Float32Array; abc: Float32Array; sigma: Float32Array; maxE2: number; minE2: number } {
	const { vols, sizes } = buildTetCache(mesh);
	const tets = mesh.tets;
	const nodes = mesh.nodes;
	const nTets = vols.length;

	// Per-tet time-averaged energy ≈ (A + B) / 2 averaged over the 4 nodes.
	const energy = new Float64Array(nTets);
	let eMax = 0;
	for (let t = 0; t < nTets; t++) {
		let sum = 0;
		for (let k = 0; k < 4; k++) {
			const nd = tets[t * 4 + k] * 3;
			sum += 0.5 * (fieldAbc[nd] + fieldAbc[nd + 1]);
		}
		const e = sum / 4;
		energy[t] = e > 0 ? e : 0;
		if (energy[t] > eMax) eMax = energy[t];
	}
	if (eMax <= 0) eMax = 1;

	// Energy-weighted CDF: weight = volume × (floor + energy_norm).
	const cdf = new Float64Array(nTets);
	let acc = 0;
	for (let t = 0; t < nTets; t++) {
		const eNorm = energy[t] / eMax;
		acc += vols[t] * (ENERGY_FLOOR + (1 - ENERGY_FLOOR) * eNorm);
		cdf[t] = acc;
	}
	const totalWeight = acc || 1;

	const positions = new Float32Array(n * 3);
	const abc = new Float32Array(n * 3);
	const sigma = new Float32Array(n);
	const w: [number, number, number, number] = [0, 0, 0, 0];
	let minE2 = Infinity, maxE2 = 0;
	for (let i = 0; i < n; i++) {
		const ti = bsearchCdf(cdf, Math.random() * totalWeight);
		uniformBary(w);
		const a = tets[ti * 4], b = tets[ti * 4 + 1],
		      c = tets[ti * 4 + 2], d = tets[ti * 4 + 3];
		positions[i * 3] = (
			w[0] * nodes[a * 3] + w[1] * nodes[b * 3] +
			w[2] * nodes[c * 3] + w[3] * nodes[d * 3]
		);
		positions[i * 3 + 1] = (
			w[0] * nodes[a * 3 + 1] + w[1] * nodes[b * 3 + 1] +
			w[2] * nodes[c * 3 + 1] + w[3] * nodes[d * 3 + 1]
		);
		positions[i * 3 + 2] = (
			w[0] * nodes[a * 3 + 2] + w[1] * nodes[b * 3 + 2] +
			w[2] * nodes[c * 3 + 2] + w[3] * nodes[d * 3 + 2]
		);
		const A = w[0] * fieldAbc[a * 3]     + w[1] * fieldAbc[b * 3]     + w[2] * fieldAbc[c * 3]     + w[3] * fieldAbc[d * 3];
		const B = w[0] * fieldAbc[a * 3 + 1] + w[1] * fieldAbc[b * 3 + 1] + w[2] * fieldAbc[c * 3 + 1] + w[3] * fieldAbc[d * 3 + 1];
		const C = w[0] * fieldAbc[a * 3 + 2] + w[1] * fieldAbc[b * 3 + 2] + w[2] * fieldAbc[c * 3 + 2] + w[3] * fieldAbc[d * 3 + 2];
		abc[i * 3] = A; abc[i * 3 + 1] = B; abc[i * 3 + 2] = C;
		sigma[i] = sizes[ti];
		const e2 = 0.5 * (A + B);
		if (e2 > 0) {
			if (e2 < minE2) minE2 = e2;
			if (e2 > maxE2) maxE2 = e2;
		}
	}
	if (!isFinite(minE2) || maxE2 === 0) { minE2 = 1; maxE2 = 1; }
	return { positions, abc, sigma, maxE2, minE2 };
}
