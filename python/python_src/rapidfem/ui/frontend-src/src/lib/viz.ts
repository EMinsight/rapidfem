/**
 * Volumetric field splat sampling.
 *
 * Identical to the old point-cloud sampler — N random points drawn across
 * the mesh volume, (A, B, C) interpolated at each by barycentric weights —
 * with one modifier: the per-tet draw probability is `volume × energy`
 * instead of `volume`, so sample density follows the field energy. Each
 * sample becomes one world-space Gaussian splat.
 *
 * Because the weighting now depends on the field, the sampling CDF is built
 * per `viz_sample` call — `viz_load_mesh` only caches the field-independent
 * per-tet volumes.
 *
 * Coefficients encode |E(t)|² = A cos²(ωt) + B sin²(ωt) − 2 C cos·sin, which
 * the splat shader composites every frame against a phase uniform.
 */
import type { MeshData } from './msh';

/** Global splat σ as a fraction of the mean sample spacing cbrt(V/n). One σ
 *  for the whole cloud (not per-tet) — 0.5 ⇒ neighbouring splats just touch
 *  on average, which keeps the cloud continuous without runaway overdraw. */
const SPACING_FACTOR = 0.5;

/** Energy coverage floor: a tet with zero field still keeps this small
 *  fraction of its volume weight, so vacuum regions don't drop out entirely
 *  and the cloud stays continuous. The dominant term is still `volume × energy`. */
const ENERGY_FLOOR = 0.08;

interface VizCache {
	nodes: Float64Array;
	tets: Uint32Array;
	vols: Float64Array;       // |tet volume|, per tet
}

let cache: VizCache | null = null;

export async function viz_load_mesh(m: MeshData): Promise<void> {
	const n_tets = m.tets.length / 4;
	if (n_tets === 0) {
		cache = null;
		return;
	}
	const vols = new Float64Array(n_tets);
	for (let t = 0; t < n_tets; t++) {
		const a = m.tets[t * 4 + 0] * 3;
		const b = m.tets[t * 4 + 1] * 3;
		const c = m.tets[t * 4 + 2] * 3;
		const d = m.tets[t * 4 + 3] * 3;
		// V = |det(b−a, c−a, d−a)| / 6
		const e1x = m.nodes[b] - m.nodes[a];
		const e1y = m.nodes[b + 1] - m.nodes[a + 1];
		const e1z = m.nodes[b + 2] - m.nodes[a + 2];
		const e2x = m.nodes[c] - m.nodes[a];
		const e2y = m.nodes[c + 1] - m.nodes[a + 1];
		const e2z = m.nodes[c + 2] - m.nodes[a + 2];
		const e3x = m.nodes[d] - m.nodes[a];
		const e3y = m.nodes[d + 1] - m.nodes[a + 1];
		const e3z = m.nodes[d + 2] - m.nodes[a + 2];
		const det =
			e1x * (e2y * e3z - e2z * e3y) -
			e1y * (e2x * e3z - e2z * e3x) +
			e1z * (e2x * e3y - e2y * e3x);
		vols[t] = Math.abs(det) / 6;
	}
	cache = { nodes: m.nodes, tets: m.tets, vols };
}

/** Binary search for the smallest index `i` with cdf[i] >= u. */
function bsearch_cdf(cdf: Float64Array, u: number): number {
	let lo = 0, hi = cdf.length - 1;
	while (lo < hi) {
		const mid = (lo + hi) >>> 1;
		if (cdf[mid] < u) lo = mid + 1;
		else hi = mid;
	}
	return lo;
}

/** Uniform barycentric weights for a tetrahedron (sorted-triple trick). */
function uniform_bary(out: [number, number, number, number]): void {
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

/**
 * Energy-weighted sampling of N field splats.
 *
 * @param field_abc  per-node [A, B, C] phasor terms
 * @param n          number of splats to draw
 */
export async function viz_sample(
	field_abc: Float32Array,
	n: number,
): Promise<{
	positions: Float32Array;
	abc: Float32Array;
	sigma: Float32Array;
	log_floor: number;
	log_range: number;
	field_range: { min: number; max: number; decades: number };
}> {
	const empty = {
		positions: new Float32Array(0),
		abc: new Float32Array(0),
		sigma: new Float32Array(0),
		log_floor: 0,
		log_range: 1,
		field_range: { min: 0, max: 1, decades: 0 },
	};
	if (!cache || n <= 0 || !field_abc || field_abc.length === 0) return empty;

	const { nodes, tets, vols } = cache;
	const n_tets = vols.length;

	// ── Per-tet time-averaged energy from the phasor terms ──────────────
	// |E|²_avg ≈ (A + B) / 2, averaged over the tet's 4 nodes.
	const energy = new Float64Array(n_tets);
	let e_max = 0;
	for (let t = 0; t < n_tets; t++) {
		let sum = 0;
		for (let k = 0; k < 4; k++) {
			const nd = tets[t * 4 + k] * 3;
			sum += 0.5 * (field_abc[nd] + field_abc[nd + 1]);
		}
		const e = sum / 4;
		energy[t] = e > 0 ? e : 0;
		if (energy[t] > e_max) e_max = energy[t];
	}
	if (e_max <= 0) e_max = 1;

	// ── Energy-weighted CDF: weight = volume × (floor + energy_norm) ────
	const cdf = new Float64Array(n_tets);
	let acc = 0;
	let total_vol = 0;
	for (let t = 0; t < n_tets; t++) {
		const e_norm = energy[t] / e_max;
		acc += vols[t] * (ENERGY_FLOOR + (1 - ENERGY_FLOOR) * e_norm);
		total_vol += vols[t];
		cdf[t] = acc;
	}
	if (acc <= 0) return empty;
	const inv_total = 1 / acc;
	for (let t = 0; t < n_tets; t++) cdf[t] *= inv_total;

	// ── One global σ from the mean sample spacing ───────────────────────
	// n points spread over total_vol sit ≈ cbrt(total_vol / n) apart; σ a
	// fraction of that keeps the cloud continuous with bounded overdraw.
	const splat_sigma = Math.cbrt(total_vol / n) * SPACING_FACTOR;

	// ── Draw N splats ───────────────────────────────────────────────────
	const positions = new Float32Array(n * 3);
	const abc = new Float32Array(n * 3);
	const sigma = new Float32Array(n);
	const w: [number, number, number, number] = [0, 0, 0, 0];
	let min_e2 = Infinity, max_e2 = 0;

	for (let i = 0; i < n; i++) {
		const t_idx = bsearch_cdf(cdf, Math.random());
		uniform_bary(w);
		const a = tets[t_idx * 4 + 0];
		const b = tets[t_idx * 4 + 1];
		const c = tets[t_idx * 4 + 2];
		const d = tets[t_idx * 4 + 3];
		positions[i * 3] =
			w[0] * nodes[a * 3]     + w[1] * nodes[b * 3]     +
			w[2] * nodes[c * 3]     + w[3] * nodes[d * 3];
		positions[i * 3 + 1] =
			w[0] * nodes[a * 3 + 1] + w[1] * nodes[b * 3 + 1] +
			w[2] * nodes[c * 3 + 1] + w[3] * nodes[d * 3 + 1];
		positions[i * 3 + 2] =
			w[0] * nodes[a * 3 + 2] + w[1] * nodes[b * 3 + 2] +
			w[2] * nodes[c * 3 + 2] + w[3] * nodes[d * 3 + 2];
		const A = w[0] * field_abc[a * 3]     + w[1] * field_abc[b * 3]     + w[2] * field_abc[c * 3]     + w[3] * field_abc[d * 3];
		const B = w[0] * field_abc[a * 3 + 1] + w[1] * field_abc[b * 3 + 1] + w[2] * field_abc[c * 3 + 1] + w[3] * field_abc[d * 3 + 1];
		const C = w[0] * field_abc[a * 3 + 2] + w[1] * field_abc[b * 3 + 2] + w[2] * field_abc[c * 3 + 2] + w[3] * field_abc[d * 3 + 2];
		abc[i * 3]     = A;
		abc[i * 3 + 1] = B;
		abc[i * 3 + 2] = C;
		sigma[i] = splat_sigma;
		// Time-averaged |E|² = (A + B) / 2.
		const e2 = 0.5 * (A + B);
		if (e2 > 0) {
			if (e2 < min_e2) min_e2 = e2;
			if (e2 > max_e2) max_e2 = e2;
		}
	}

	if (!isFinite(min_e2) || max_e2 === 0) {
		min_e2 = 1; max_e2 = 1;
	}
	// Shader works in log10(|E|), not log10(|E|²). Convert.
	const e_min_raw = Math.sqrt(Math.max(min_e2, 0));
	const e_max_v = Math.sqrt(max_e2);
	const e_min = Math.max(e_min_raw, e_max_v * 1e-3);
	const log_max = Math.log10(e_max_v);
	const log_min = Math.log10(e_min);
	const log_floor = log_min;
	const log_range = Math.max(log_max - log_min, 0.5);
	const decades = log_max - log_min;
	return {
		positions,
		abc,
		sigma,
		log_floor,
		log_range,
		field_range: { min: e_min, max: e_max_v, decades },
	};
}
