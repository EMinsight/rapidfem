/**
 * Volumetric field point-cloud sampling.
 *
 * N random points drawn across the mesh volume with (A, B, C) phasor
 * coefficients interpolated by barycentric weights. Per-node density driver
 *     w(x) = ENERGY_FLOOR + (1 − ENERGY_FLOOR) · e(x) / e_ref,
 * with e(x) = 0.5·(A(x)+B(x)) the time-averaged |E|² and e_ref the
 * percentile-clipped energy max (top-1% outliers don't squash the bias).
 *
 * Two-stage draw:
 *   1. Pick tet ∝ vol[t]^VOL_ALPHA · w_max[t], w_max = max of w over corners.
 *   2. Inside the tet: uniform barycentric proposal, accept with
 *      prob = w(x) / w_max[t].
 *
 * With VOL_ALPHA = 1 the marginal density per unit volume reduces to
 * exactly ρ(x) = w(x), so there is no leak of tet-size variation into
 * visible brightness. Small-tet starvation in big-air-dominated meshes
 * (Microstrip, Coax, Inductor) is handled by the energy bias instead:
 * with ENERGY_FLOOR = 0.2 and percentile-clipped e_ref, hot tets get a
 * 5× per-volume sample-count uplift regardless of their size.
 *
 * Rejection inside the tet keeps density piecewise-LINEAR across faces
 * (vs piecewise-CONSTANT per-tet mean), so visible "tet edge" density
 * jumps stay killed.
 *
 * Because the weighting depends on the field, the CDF is built per
 * `viz_sample` call — `viz_load_mesh` only caches the field-independent
 * per-tet volumes.
 *
 * Coefficients encode |E(t)|² = A cos²(ωt) + B sin²(ωt) − 2 C cos·sin, which
 * the point shader composites every frame against a phase uniform.
 */
import type { MeshData } from './msh';

/** Mixing weight between uniform spatial coverage and energy-following
 *  density. Effective per-unit-volume density is
 *      w(x) = ENERGY_FLOOR + (1 − ENERGY_FLOOR) · e(x) / e_ref,
 *  where e_ref is the percentile-clipped energy max (see ENERGY_PERCENTILE).
 *  Floor 1.0 = perfectly uniform; floor 0.0 = vacuum gets zero samples
 *  (strong bias). Default sits low (strong energy following) so structure-
 *  bearing tets dominate sample count over big empty air tets.
 *  Caveat: in non-radiating channels like J, vacuum samples carry e = 0
 *  and show up as the lowest colormap value (visible but dim). */
const ENERGY_FLOOR = 0.2;

/** Volume exponent in the tet-pick CDF. Tet probability ∝ vol[t]^α · w_max[t].
 *  Kept at 1.0 because the marginal density per unit volume then reduces to
 *  w(x) exactly — perfectly smooth across the mesh, no leak of tet-size
 *  variation into visible brightness. α<1 (an earlier attempt to equalise
 *  sample COUNT across tets) inverted into the opposite artifact: small
 *  tets at grading boundaries ended up with higher density per volume than
 *  their neighbours and lit up as bright "pockets" along otherwise smooth
 *  fields. Small-tet starvation in big-air-dominated meshes is fixed by the
 *  energy bias instead (low ENERGY_FLOOR + percentile-clipped reference),
 *  which gives hot tets ~1/ENERGY_FLOOR × the per-volume sample count
 *  regardless of their size. */
const VOL_ALPHA = 1.0;

/** Reference for normalising e(x) in the density weight. Using the true max
 *  squashes the bias signal across the whole mesh: a single port-edge node
 *  is typically 1000× stronger than the structure-following bulk, so
 *  e/e_max ≈ 0 everywhere except the port and the bias goes flat. Clipping
 *  e_ref to a percentile (here 99%) sets the reference at the body of the
 *  high-energy distribution, not the single outlier, so the conductor /
 *  microstrip line sits in the bias-active range. */
const ENERGY_PERCENTILE = 0.99;

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

/** Percentile of the positive entries of `arr`, computed via a single-pass
 *  bit-pattern histogram on IEEE 754 floats (the top 11 bits of a positive
 *  float are monotonic with its value, so binning them is a free log-bin).
 *  O(n) instead of O(n log n) sort — matters at 100k+ nodes. */
function positive_percentile(arr: Float64Array, p: number): number {
	const N_BINS = 2048;
	const counts = new Uint32Array(N_BINS);
	const view_f32 = new Float32Array(1);
	const view_u32 = new Uint32Array(view_f32.buffer);
	let n_pos = 0;
	let max_val = 0;
	for (let i = 0; i < arr.length; i++) {
		const v = arr[i];
		if (v <= 0) continue;
		n_pos++;
		if (v > max_val) max_val = v;
		view_f32[0] = v;
		const bin = (view_u32[0] >>> 20) & (N_BINS - 1);
		counts[bin]++;
	}
	if (n_pos === 0) return 0;
	const target = Math.ceil(n_pos * p);
	let cum = 0;
	for (let b = 0; b < N_BINS; b++) {
		cum += counts[b];
		if (cum >= target) {
			// Lower edge of bin: reconstruct the float whose top bits give `b`.
			view_u32[0] = b << 20;
			return Math.min(view_f32[0], max_val);
		}
	}
	return max_val;
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
	// Clipped HERE at the percentile, not just normalised against it. The
	// P1 edge-element projection produces node-level energy spikes at
	// sliver-tet corners (the projection is one-sided and not smooth across
	// fine-coarse interfaces), and an unclipped corner spike pins ONE tet
	// to w_max = 1 in the CDF — that tet then collects a disproportionate
	// pile of samples, visible as "pockets" of bright noise inside an
	// otherwise smoothly-decaying coax field. Clipping the per-node values
	// against e_ref BEFORE w_max_tet kills the preferential picking; the
	// in-tet rejection still recovers a piecewise-linear density inside each
	// tet because the clip is per-node monotone.
	const e_node = new Float64Array(n_nodes);
	for (let i = 0; i < n_nodes; i++) {
		const e = 0.5 * (field_abc[i * 3] + field_abc[i * 3 + 1]);
		e_node[i] = e > 0 ? e : 0;
	}
	const e_ref = positive_percentile(e_node, ENERGY_PERCENTILE);
	if (e_ref <= 0) return empty;
	for (let i = 0; i < n_nodes; i++) {
		if (e_node[i] > e_ref) e_node[i] = e_ref;
	}
	const inv_e_ref = 1 / e_ref;

	// ── Per-tet max weight (upper bound for rejection inside the tet) ───
	// w(x) = ENERGY_FLOOR + (1−ENERGY_FLOOR)·e(x)/e_ref is affine over the
	// (already-clipped) e, so the per-tet max sits at the corner with the
	// largest clipped e.
	const w_max_tet = new Float64Array(n_tets);
	for (let t = 0; t < n_tets; t++) {
		let em = 0;
		for (let k = 0; k < 4; k++) {
			const en = e_node[tets[t * 4 + k]];
			if (en > em) em = en;
		}
		w_max_tet[t] = ENERGY_FLOOR + (1 - ENERGY_FLOOR) * em * inv_e_ref;
	}

	// ── Tet-pick CDF: weight = vol^VOL_ALPHA · w_max_tet ────────────────
	// VOL_ALPHA<1 stops big air tets from monopolising samples when the
	// mesh has 100-1000× size variation, so small structure-bearing tets
	// near conductors still get a fair share of points.
	const cdf = new Float64Array(n_tets);
	let acc = 0;
	for (let t = 0; t < n_tets; t++) {
		acc += Math.pow(vols[t], VOL_ALPHA) * w_max_tet[t];
		cdf[t] = acc;
	}
	if (acc <= 0) return empty;
	const inv_total = 1 / acc;
	for (let t = 0; t < n_tets; t++) cdf[t] *= inv_total;

	// ── Draw N points with rejection inside the picked tet ──────────────
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

		// e_local ≤ e_ref since corner values are pre-clipped, so the
		// saturation is implicit and no per-iteration min() is needed.
		for (let attempt = 0; attempt < MAX_PROPOSALS; attempt++) {
			uniform_bary(w);
			const e_local = w[0] * ea + w[1] * eb + w[2] * ec_ + w[3] * ed;
			const w_local = ENERGY_FLOOR + (1 - ENERGY_FLOOR) * e_local * inv_e_ref;
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

/**
 * Energy-weighted sampling of N points from the cached mesh against a
 * per-node *scalar* weight (the time-domain trajectory path).
 *
 * Reuses the exact tet-pick CDF + in-tet rejection from `viz_sample`, but
 * the weight is the supplied per-node scalar instead of `e_node`, and the
 * sample is *static*: rather than evaluating an (A,B,C) phasor, it records
 * the picked tet index and the 4 barycentric weights so the cheap
 * `viz_eval_static` can interpolate any per-frame field at those fixed
 * sample points.
 *
 * @param weight  per-node scalar density driver (e.g. peak |E| per node)
 * @param n       number of points to draw
 */
export async function viz_sample_static(
	weight: Float32Array,
	n: number,
): Promise<{ positions: Float32Array; tet: Uint32Array; bary: Float32Array }> {
	const empty = {
		positions: new Float32Array(0),
		tet: new Uint32Array(0),
		bary: new Float32Array(0),
	};
	if (!cache || n <= 0 || !weight || weight.length === 0) return empty;

	const { nodes, tets, vols } = cache;
	const n_tets = vols.length;
	const n_nodes = weight.length;

	// ── Per-node non-negative density driver, percentile-clipped ────────
	const e_node = new Float64Array(n_nodes);
	for (let i = 0; i < n_nodes; i++) {
		e_node[i] = weight[i] > 0 ? weight[i] : 0;
	}
	const e_ref = positive_percentile(e_node, ENERGY_PERCENTILE);
	if (e_ref <= 0) return empty;
	for (let i = 0; i < n_nodes; i++) {
		if (e_node[i] > e_ref) e_node[i] = e_ref;
	}
	const inv_e_ref = 1 / e_ref;

	// ── Per-tet max weight (upper bound for rejection inside the tet) ───
	const w_max_tet = new Float64Array(n_tets);
	for (let t = 0; t < n_tets; t++) {
		let em = 0;
		for (let k = 0; k < 4; k++) {
			const en = e_node[tets[t * 4 + k]];
			if (en > em) em = en;
		}
		w_max_tet[t] = ENERGY_FLOOR + (1 - ENERGY_FLOOR) * em * inv_e_ref;
	}

	// ── Tet-pick CDF: weight = vol^VOL_ALPHA · w_max_tet ────────────────
	const cdf = new Float64Array(n_tets);
	let acc = 0;
	for (let t = 0; t < n_tets; t++) {
		acc += Math.pow(vols[t], VOL_ALPHA) * w_max_tet[t];
		cdf[t] = acc;
	}
	if (acc <= 0) return empty;
	const inv_total = 1 / acc;
	for (let t = 0; t < n_tets; t++) cdf[t] *= inv_total;

	// ── Draw N points with rejection inside the picked tet ──────────────
	const MAX_PROPOSALS = 64;
	const positions = new Float32Array(n * 3);
	const tet = new Uint32Array(n);
	const bary = new Float32Array(n * 4);
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
			const w_local = ENERGY_FLOOR + (1 - ENERGY_FLOOR) * e_local * inv_e_ref;
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
		tet[i] = t_idx;
		bary[i * 4]     = w[0];
		bary[i * 4 + 1] = w[1];
		bary[i * 4 + 2] = w[2];
		bary[i * 4 + 3] = w[3];
	}

	return { positions, tet, bary };
}

/**
 * Cheap per-frame evaluator for a static sample (`viz_sample_static`).
 *
 * For each point `i`, the magnitude is the barycentric mix of the
 * per-node `field` over the 4 corners of the recorded tet:
 *     mag[i] = Σ_k bary[i·4+k] · field[ cache.tets[tet[i]·4+k] ].
 *
 * @param field  per-node scalar field for one frame
 * @param tet    per-point tet index   (from `viz_sample_static`)
 * @param bary   per-point 4 bary weights (from `viz_sample_static`)
 */
export function viz_eval_static(
	field: Float32Array,
	tet: Uint32Array,
	bary: Float32Array,
): Float32Array {
	const n = tet.length;
	const out = new Float32Array(n);
	if (!cache || n === 0) return out;
	const { tets } = cache;
	for (let i = 0; i < n; i++) {
		const base = tet[i] * 4;
		out[i] =
			bary[i * 4]     * field[tets[base]]     +
			bary[i * 4 + 1] * field[tets[base + 1]] +
			bary[i * 4 + 2] * field[tets[base + 2]] +
			bary[i * 4 + 3] * field[tets[base + 3]];
	}
	return out;
}

/**
 * Colour range (min/max) from a per-node *scalar* field — the static-field
 * analogue of `field_energy_range`. Clips the auto-range to the
 * RANGE_PERCENTILE band so a handful of outlier nodes don't dominate.
 */
export function viz_scalar_range(field: Float32Array): {
	min: number;
	max: number;
} {
	const vals: number[] = [];
	for (let i = 0; i < field.length; i++) {
		if (field[i] > 0) vals.push(field[i]);
	}
	if (vals.length === 0) return { min: 0, max: 1 };
	vals.sort((a, b) => a - b);
	const lo_idx = Math.min(vals.length - 1, Math.floor(vals.length * (1 - RANGE_PERCENTILE)));
	const hi_idx = Math.min(vals.length - 1, Math.floor(vals.length * RANGE_PERCENTILE));
	const max = vals[hi_idx];
	const min = Math.max(vals[lo_idx], max * 1e-3);
	return { min, max };
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
