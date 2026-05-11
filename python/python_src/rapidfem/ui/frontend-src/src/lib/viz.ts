/**
 * Volumetric field point-cloud sampling.
 *
 * Replaces the WASM worker that used to do this. Given a mesh + per-node
 * (A, B, C) phasor coefficients, we draw N random points uniformly within
 * the tet volume (weighted by per-tet volume so density is uniform) and
 * interpolate (A, B, C) at each point via barycentric weights. Coefficients
 * encode the time-domain |E(t)|² as A cos²(ωt) + B sin²(ωt) − 2 C cos·sin,
 * which the GPU shader composites every frame against a phase uniform.
 */
import type { MeshData } from './msh';

interface VizCache {
	nodes: Float64Array;
	tets: Uint32Array;
	cdf: Float64Array;        // tet-volume CDF (length = n_tets)
	total_vol: number;
}

let cache: VizCache | null = null;

export async function viz_load_mesh(m: MeshData): Promise<void> {
	const n_tets = m.tets.length / 4;
	if (n_tets === 0) {
		cache = null;
		return;
	}
	const vols = new Float64Array(n_tets);
	let total = 0;
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
		const v = Math.abs(det) / 6;
		vols[t] = v;
		total += v;
	}
	const cdf = new Float64Array(n_tets);
	let acc = 0;
	for (let t = 0; t < n_tets; t++) {
		acc += vols[t];
		cdf[t] = acc / total;
	}
	cache = {
		nodes: m.nodes,
		tets: m.tets,
		cdf,
		total_vol: total,
	};
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
	// Sort ascending
	if (s > t) [s, t] = [t, s];
	if (t > u) [t, u] = [u, t];
	if (s > t) [s, t] = [t, s];
	out[0] = s;
	out[1] = t - s;
	out[2] = u - t;
	out[3] = 1 - u;
}

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
	if (!cache || n <= 0 || !field_abc || field_abc.length === 0) {
		return {
			positions: new Float32Array(0),
			abc: new Float32Array(0),
			log_floor: 0,
			log_range: 1,
			field_range: { min: 0, max: 1, decades: 0 },
		};
	}
	const { nodes, tets, cdf } = cache;
	const positions = new Float32Array(n * 3);
	const abc = new Float32Array(n * 3);
	const w: [number, number, number, number] = [0, 0, 0, 0];

	// Track time-averaged |E|² ≈ (A + B) / 2 for log range.
	let min_e2 = Infinity, max_e2 = 0;

	for (let i = 0; i < n; i++) {
		const u = Math.random();
		const t_idx = bsearch_cdf(cdf, u);
		uniform_bary(w);
		const a = tets[t_idx * 4 + 0];
		const b = tets[t_idx * 4 + 1];
		const c = tets[t_idx * 4 + 2];
		const d = tets[t_idx * 4 + 3];
		const px =
			w[0] * nodes[a * 3]     + w[1] * nodes[b * 3]     +
			w[2] * nodes[c * 3]     + w[3] * nodes[d * 3];
		const py =
			w[0] * nodes[a * 3 + 1] + w[1] * nodes[b * 3 + 1] +
			w[2] * nodes[c * 3 + 1] + w[3] * nodes[d * 3 + 1];
		const pz =
			w[0] * nodes[a * 3 + 2] + w[1] * nodes[b * 3 + 2] +
			w[2] * nodes[c * 3 + 2] + w[3] * nodes[d * 3 + 2];
		positions[i * 3]     = px;
		positions[i * 3 + 1] = py;
		positions[i * 3 + 2] = pz;
		const aA = field_abc[a * 3], aB = field_abc[a * 3 + 1], aC = field_abc[a * 3 + 2];
		const bA = field_abc[b * 3], bB = field_abc[b * 3 + 1], bC = field_abc[b * 3 + 2];
		const cA = field_abc[c * 3], cB = field_abc[c * 3 + 1], cC = field_abc[c * 3 + 2];
		const dA = field_abc[d * 3], dB = field_abc[d * 3 + 1], dC = field_abc[d * 3 + 2];
		const A = w[0] * aA + w[1] * bA + w[2] * cA + w[3] * dA;
		const B = w[0] * aB + w[1] * bB + w[2] * cB + w[3] * dB;
		const C = w[0] * aC + w[1] * bC + w[2] * cC + w[3] * dC;
		abc[i * 3]     = A;
		abc[i * 3 + 1] = B;
		abc[i * 3 + 2] = C;
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
	const log_max = Math.log10(max_e2);
	const log_min = Math.log10(Math.max(min_e2, max_e2 * 1e-6));
	const log_floor = log_min;
	const log_range = Math.max(log_max - log_min, 1);
	// |E| = sqrt(|E|²)
	const e_min = Math.sqrt(Math.max(min_e2, 0));
	const e_max = Math.sqrt(max_e2);
	const decades = Math.log10(e_max / Math.max(e_min, e_max * 1e-6));
	return {
		positions,
		abc,
		log_floor,
		log_range,
		field_range: { min: e_min, max: e_max, decades },
	};
}
