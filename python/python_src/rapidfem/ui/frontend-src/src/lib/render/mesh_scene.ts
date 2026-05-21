/**
 * Shared mesh-to-render helpers. Used by both the in-app MeshViewer
 * component and the standalone <fem-viewer> web-component embed so the
 * two share the SAME geometry pipeline — same per-volume hull
 * extraction, same outward-facing face normals, same float64 cross
 * product. The visible artefacts you only get when these differ
 * (dappled shading on flat faces, random face flips) bit us once;
 * keep them centralised here.
 */

/** Subset of the mesh payload everyone needs. */
export interface MeshLike {
	nodes: number[] | Float32Array | Float64Array;
	tris: number[];
	tets: number[];
	tet_phys: number[];
}

// ── Per-volume outer hull ─────────────────────────────────────────────
//
// For each physical-group volume independently: a tet face appearing
// exactly once = boundary of that volume. Internal faces (shared with
// another tet of the same volume) cancel out.
//
// CRITICAL: every boundary triangle is oriented so its face normal
// points AWAY from the tet's fourth vertex (= outward from the volume).
// Without that, adjacent boundary triangles can have flipped normals
// → dappled shading on flat surfaces.

export function buildVolumeBoundaries(m: MeshLike): Map<number, number[]> {
	const enc = (a: number, b: number, c: number): bigint => {
		const s = [a, b, c].sort((x, y) => x - y);
		return (BigInt(s[0]) * 0x100000000n + BigInt(s[1])) * 0x100000000n + BigInt(s[2]);
	};
	const per_vol = new Map<number, number[]>();
	const ntets = m.tet_phys.length;
	for (let t = 0; t < ntets; t++) {
		const v = m.tet_phys[t];
		if (!v) continue;
		let arr = per_vol.get(v);
		if (!arr) { arr = []; per_vol.set(v, arr); }
		arr.push(t);
	}

	const orient_outward = (
		a: number, b: number, c: number, o: number,
	): [number, number, number] => {
		const ax = m.nodes[a * 3], ay = m.nodes[a * 3 + 1], az = m.nodes[a * 3 + 2];
		const bx = m.nodes[b * 3], by = m.nodes[b * 3 + 1], bz = m.nodes[b * 3 + 2];
		const cx = m.nodes[c * 3], cy = m.nodes[c * 3 + 1], cz = m.nodes[c * 3 + 2];
		const ox = m.nodes[o * 3], oy = m.nodes[o * 3 + 1], oz = m.nodes[o * 3 + 2];
		const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
		const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
		const nx = e1y * e2z - e1z * e2y;
		const ny = e1z * e2x - e1x * e2z;
		const nz = e1x * e2y - e1y * e2x;
		const dx = ox - ax, dy = oy - ay, dz = oz - az;
		// If the face normal points toward the opposite vertex `o`, the
		// triangle is wound inward — swap b/c to flip it.
		if (nx * dx + ny * dy + nz * dz > 0) return [a, c, b];
		return [a, b, c];
	};

	const out = new Map<number, number[]>();
	for (const [vol, tet_indices] of per_vol.entries()) {
		const seen = new Map<bigint, { count: number; tri: [number, number, number] }>();
		for (const t of tet_indices) {
			const a = m.tets[t * 4], b = m.tets[t * 4 + 1],
			      c = m.tets[t * 4 + 2], d = m.tets[t * 4 + 3];
			const tri_descs: [[number, number, number], number][] = [
				[[a, b, c], d],
				[[a, b, d], c],
				[[a, c, d], b],
				[[b, c, d], a],
			];
			for (const [f, opp] of tri_descs) {
				const k = enc(f[0], f[1], f[2]);
				const prev = seen.get(k);
				if (!prev) {
					seen.set(k, { count: 1, tri: orient_outward(f[0], f[1], f[2], opp) });
				} else {
					prev.count++;
				}
			}
		}
		const arr: number[] = [];
		for (const e of seen.values()) {
			if (e.count === 1) arr.push(e.tri[0], e.tri[1], e.tri[2]);
		}
		if (arr.length) out.set(vol, arr);
	}
	return out;
}
