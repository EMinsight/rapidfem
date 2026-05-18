/**
 * Volumetric field point-cloud sampling.
 *
 * N random points drawn across the mesh volume with (A, B, C) phasor
 * coefficients interpolated by barycentric weights. Target sample density
 * per unit volume is the linear-interpolated weight
 *     w(x) = ENERGY_FLOOR + (1 − ENERGY_FLOOR) · e(x)/e_global_max,
 * where e(x) = 0.5·(A(x)+B(x)) is the time-averaged |E|².
 *
 * Two-stage draw to make the density piecewise-LINEAR across the volume
 * (not piecewise-constant as in the old per-tet-mean weighting):
 *   1. Pick tet ∝ vol[t] · w_max[t]  with  w_max[t] = max over the 4 corners.
 *   2. Inside the tet: uniform barycentric proposal, accept with
 *      prob = w(x) / w_max[t].
 * Marginal density per unit volume reduces to w(x) exactly.
 *
 * Why this matters: with the old per-tet-mean weight, the point count per
 * tet was piecewise-constant → visible density jumps at shared faces (the
 * "tet edges" artifact). Rejection inside the tet gives a density that
 * agrees on both sides of a shared face because the 3 face nodes evaluate
 * to the same linear value from either tet.
 *
 * Because the weighting depends on the field, the CDF is built per
 * `viz_sample` call — `viz_load_mesh` only caches the field-independent
 * per-tet volumes.
 *
 * Coefficients encode |E(t)|² = A cos²(ωt) + B sin²(ωt) − 2 C cos·sin, which
 * the point shader composites every frame against a phase uniform.
 */
import type { MeshData } from './msh';

/** Energy coverage floor: a tet with zero field gets exactly zero sampling
 *  weight when this is 0. We want that: for the J channel, σ_eff = 0 in air
 *  → J = 0 in air → no point in placing samples there. Same logic for any
 *  channel — show what's there, don't pad vacuum. */
const ENERGY_FLOOR = 0.0;

/** Colormap upper percentile. The auto-range is dominated by a handful of
 *  outlier nodes — typically the driven-port edges where the imposed E (and
 *  therefore curl E → H) is far stronger than anywhere in the bulk. Clipping
 *  at the 99th percentile lets the bulk use the full colour range while the
 *  port saturates at the top. */
const RANGE_PERCENTILE = 0.99;

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
	log_floor: number;
	log_range: number;
	field_range: { min: number; max: number; decades: number };
}> {
	const empty = {
		positions: new Float32Array(0),
		abc: new Float32Array(0),
		log_floor: 0,
		log_range: 1,
		field_range: { min: 0, max: 1, decades: 0 },
	};
	if (!cache || n <= 0 || !field_abc || field_abc.length === 0) return empty;

	const { nodes, tets, vols } = cache;
	const n_tets = vols.length;
	const n_nodes = field_abc.length / 3;

	// ── Per-node time-averaged energy e_i = 0.5·(A_i + B_i) ─────────────
	const e_node = new Float64Array(n_nodes);
	let e_global_max = 0;
	for (let i = 0; i < n_nodes; i++) {
		const e = 0.5 * (field_abc[i * 3] + field_abc[i * 3 + 1]);
		const ec = e > 0 ? e : 0;
		e_node[i] = ec;
		if (ec > e_global_max) e_global_max = ec;
	}
	if (e_global_max <= 0) return empty;
	const inv_e_global = 1 / e_global_max;

	// ── Per-tet max weight (upper bound for rejection inside the tet) ───
	// w(x) = ENERGY_FLOOR + (1−ENERGY_FLOOR)·e(x)/e_global_max is affine in
	// the barycentric coords, so w_max_in_tet = w at the corner with
	// max e_node.
	const w_max_tet = new Float64Array(n_tets);
	for (let t = 0; t < n_tets; t++) {
		let em = 0;
		for (let k = 0; k < 4; k++) {
			const en = e_node[tets[t * 4 + k]];
			if (en > em) em = en;
		}
		w_max_tet[t] = ENERGY_FLOOR + (1 - ENERGY_FLOOR) * em * inv_e_global;
	}

	// ── Tet-pick CDF: weight = vol · w_max_tet  (so the marginal density
	//    per unit volume after rejection is exactly w(x)) ────────────────
	const cdf = new Float64Array(n_tets);
	let acc = 0;
	for (let t = 0; t < n_tets; t++) {
		acc += vols[t] * w_max_tet[t];
		cdf[t] = acc;
	}
	if (acc <= 0) return empty;
	const inv_total = 1 / acc;
	for (let t = 0; t < n_tets; t++) cdf[t] *= inv_total;

	// ── Draw N points with rejection inside the picked tet ──────────────
	// Rejection budget: w_max_tet caps acceptance from below by 1/4
	// (energy concentrated in one corner of a 4-node element), so the
	// expected proposal count per sample is at most ~4. Cap iterations
	// to keep numerical edge cases bounded.
	const MAX_PROPOSALS = 64;
	const positions = new Float32Array(n * 3);
	const abc = new Float32Array(n * 3);
	const w: [number, number, number, number] = [0, 0, 0, 0];
	for (let i = 0; i < n; i++) {
		const t_idx = bsearch_cdf(cdf, Math.random());
		const a = tets[t_idx * 4 + 0];
		const b = tets[t_idx * 4 + 1];
		const c = tets[t_idx * 4 + 2];
		const d = tets[t_idx * 4 + 3];
		const ea = e_node[a], eb = e_node[b], ec_ = e_node[c], ed = e_node[d];
		const wmax = w_max_tet[t_idx];

		for (let attempt = 0; attempt < MAX_PROPOSALS; attempt++) {
			uniform_bary(w);
			const e_local = w[0] * ea + w[1] * eb + w[2] * ec_ + w[3] * ed;
			const w_local = ENERGY_FLOOR + (1 - ENERGY_FLOOR) * e_local * inv_e_global;
			if (Math.random() * wmax <= w_local) break;
		}

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
	}

	// Colormap range from the full per-node energy distribution (not from the
	// energy-biased samples). The 99th percentile clips port-driven outliers
	// so the bulk can use the full colour range; the small-end floor is the
	// 1st percentile for symmetry. |E|²_avg = (A + B) / 2 per node.
	const range = field_energy_range(field_abc);
	return {
		positions,
		abc,
		log_floor: range.log_floor,
		log_range: range.log_range,
		field_range: range.field_range,
	};
}

function field_energy_range(field_abc: Float32Array): {
	log_floor: number;
	log_range: number;
	field_range: { min: number; max: number; decades: number };
} {
	const n_nodes = field_abc.length / 3;
	const energies: number[] = [];
	for (let ni = 0; ni < n_nodes; ni++) {
		const e2 = 0.5 * (field_abc[ni * 3] + field_abc[ni * 3 + 1]);
		if (e2 > 0) energies.push(e2);
	}
	if (energies.length === 0) {
		return { log_floor: 0, log_range: 1, field_range: { min: 1, max: 1, decades: 0 } };
	}
	energies.sort((a, b) => a - b);
	const lo_idx = Math.min(energies.length - 1, Math.floor(energies.length * (1 - RANGE_PERCENTILE)));
	const hi_idx = Math.min(energies.length - 1, Math.floor(energies.length * RANGE_PERCENTILE));
	const e2_lo = energies[lo_idx];
	const e2_hi = energies[hi_idx];
	// Shader works in log10(|F|), not log10(|F|²). Convert.
	const e_max = Math.sqrt(e2_hi);
	const e_min = Math.max(Math.sqrt(e2_lo), e_max * 1e-3);
	const log_max = Math.log10(e_max);
	const log_min = Math.log10(e_min);
	return {
		log_floor: log_min,
		log_range: Math.max(log_max - log_min, 0.5),
		field_range: { min: e_min, max: e_max, decades: log_max - log_min },
	};
}
