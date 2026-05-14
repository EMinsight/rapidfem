/**
 * Depth-sort worker for the volumetric field splat cloud.
 *
 * Gaussian splats must be drawn back-to-front for correct "over" alpha
 * compositing. Doing that sort on the main thread would stutter the orbit
 * interaction at 500k+ splats, so it lives here.
 *
 * The sort is a 16-bit counting sort on the view-space depth — O(n), no
 * comparisons — which is what the fast web 3DGS renderers (antimatter15/splat,
 * kishimisu) use. Frame-to-frame the camera barely moves so the depth order
 * is nearly stable anyway; the counting sort just doesn't care either way.
 *
 * Protocol (main → worker):
 *   { type: 'load', positions: Float32Array }   — set once per cloud (transferred in)
 *   { type: 'sort', view: [x,y,z], token }      — request a sort for this view dir
 * Protocol (worker → main):
 *   { type: 'sorted', index: Uint32Array, token } — draw order, farthest-first (transferred out)
 */

const BUCKETS = 65536; // 16-bit depth quantisation

let positions: Float32Array | null = null;
let count = 0;

// Scratch buffers, grown lazily and reused across sorts.
let depthKey = new Uint32Array(0);
let counts = new Uint32Array(BUCKETS);

interface LoadMsg {
	type: 'load';
	positions: Float32Array;
}
interface SortMsg {
	type: 'sort';
	view: [number, number, number];
	token: number;
}
type InMsg = LoadMsg | SortMsg;

self.onmessage = (e: MessageEvent<InMsg>) => {
	const msg = e.data;
	if (msg.type === 'load') {
		positions = msg.positions;
		count = positions.length / 3;
		if (depthKey.length !== count) depthKey = new Uint32Array(count);
		return;
	}
	if (msg.type === 'sort') {
		if (!positions || count === 0) {
			self.postMessage({ type: 'sorted', index: new Uint32Array(0), token: msg.token });
			return;
		}
		const index = sortByDepth(positions, count, msg.view);
		(self as unknown as Worker).postMessage(
			{ type: 'sorted', index, token: msg.token },
			[index.buffer],
		);
	}
};

/** 16-bit counting sort of splat indices by view-space depth, farthest-first. */
function sortByDepth(
	pos: Float32Array,
	n: number,
	view: [number, number, number],
): Uint32Array {
	const [vx, vy, vz] = view;

	// Pass 1: project each splat onto the view direction, track the range.
	let dMin = Infinity;
	let dMax = -Infinity;
	for (let i = 0; i < n; i++) {
		const d = pos[i * 3] * vx + pos[i * 3 + 1] * vy + pos[i * 3 + 2] * vz;
		depthKey[i] = 0; // placeholder; filled below once range is known
		if (d < dMin) dMin = d;
		if (d > dMax) dMax = d;
	}
	const span = dMax - dMin || 1;
	const scale = (BUCKETS - 1) / span;

	// Pass 2: quantise to a bucket and histogram. Larger depth = farther from
	// the camera; we invert the bucket so farthest lands in bucket 0.
	counts.fill(0);
	for (let i = 0; i < n; i++) {
		const d = pos[i * 3] * vx + pos[i * 3 + 1] * vy + pos[i * 3 + 2] * vz;
		let b = (BUCKETS - 1) - (((d - dMin) * scale) | 0);
		if (b < 0) b = 0;
		else if (b >= BUCKETS) b = BUCKETS - 1;
		depthKey[i] = b;
		counts[b]++;
	}

	// Prefix sum → bucket start offsets.
	let acc = 0;
	for (let b = 0; b < BUCKETS; b++) {
		const c = counts[b];
		counts[b] = acc;
		acc += c;
	}

	// Scatter into the output order.
	const index = new Uint32Array(n);
	for (let i = 0; i < n; i++) {
		const b = depthKey[i];
		index[counts[b]++] = i;
	}
	return index;
}
