/** Minimal Gmsh MSH 4.1 reader.
 *
 * Returns the bits we need for in-browser 3D viz: nodes, surface triangles,
 * and a physical-group → name lookup. Volume tetrahedra are kept too so the
 * field viewer can sample inside them later.
 *
 * Spec: https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
 */

export interface MeshData {
	nodes: Float64Array;        // [x0,y0,z0, ...] in METERS — kept f64 for clean
	                             // analytical normals on coplanar triangles
	                             // (μm-scale geometry suffers from f32 cross-product noise)
	tris: Uint32Array;
	tri_phys: Int32Array;
	tets: Uint32Array;
	tet_phys: Int32Array;
	phys_names: Map<number, string>;
	phys_dim: Map<number, number>;
	bbox: { min: [number, number, number]; max: [number, number, number] };
}

/** Read a section delimited by $Section ... $EndSection from a line iterator. */
function* lines_of(text: string): Generator<string> {
	let start = 0;
	const n = text.length;
	for (let i = 0; i < n; i++) {
		if (text.charCodeAt(i) === 10 /* \n */) {
			yield text.slice(start, i).replace(/\r$/, '');
			start = i + 1;
		}
	}
	if (start < n) yield text.slice(start).replace(/\r$/, '');
}

export function parse_msh(text: string): MeshData {
	const it = lines_of(text);
	const next = () => {
		const r = it.next();
		return r.done ? null : r.value;
	};
	const expect = (s: string) => {
		const l = next();
		if (l !== s) throw new Error(`MSH parse: expected ${s}, got ${l}`);
	};

	let format_line: string | null = null;

	const phys_names = new Map<number, string>();
	const phys_dim = new Map<number, number>();

	// Entity tag (per dim) → physical group tag(s)
	// dim → entity_tag → first physical_tag (we use only one for color)
	const entity_phys: Map<number, Map<number, number[]>> = new Map([
		[0, new Map()],
		[1, new Map()],
		[2, new Map()],
		[3, new Map()]
	]);

	const node_idx_to_pos = new Map<number, [number, number, number]>(); // gmsh tag → xyz
	const tris_raw: number[] = [];
	const tri_phys_raw: number[] = [];
	const tets_raw: number[] = [];
	const tet_phys_raw: number[] = [];

	while (true) {
		const line = next();
		if (line == null) break;
		const trimmed = line.trim();
		if (!trimmed.startsWith('$')) continue;
		if (trimmed === '$MeshFormat') {
			format_line = next();
			expect('$EndMeshFormat');
			if (!format_line || !format_line.startsWith('4.1')) {
				throw new Error(`MSH: only format 4.1 is supported, got "${format_line}"`);
			}
		} else if (trimmed === '$PhysicalNames') {
			const n = parseInt(next() ?? '0', 10);
			for (let i = 0; i < n; i++) {
				const parts = (next() ?? '').split(' ');
				const dim = parseInt(parts[0], 10);
				const tag = parseInt(parts[1], 10);
				const name = parts.slice(2).join(' ').replace(/^"|"$/g, '');
				phys_names.set(tag, name);
				phys_dim.set(tag, dim);
			}
			expect('$EndPhysicalNames');
		} else if (trimmed === '$Entities') {
			const hdr = (next() ?? '').split(' ').map((s) => parseInt(s, 10));
			const [nPts, nCurves, nSurfs, nVols] = hdr;
			// Skip points
			for (let i = 0; i < nPts; i++) next();
			// Each entity row format (dim>0):
			//   tag minX minY minZ maxX maxY maxZ numPhysical [physTag...] numBoundary [...]
			// We extract entity_tag → physical_tags for dims 1,2,3.
			const eat_entities = (count: number, dim: number) => {
				const map = entity_phys.get(dim)!;
				for (let i = 0; i < count; i++) {
					const parts = (next() ?? '').split(/\s+/).filter((s) => s.length);
					const tag = parseInt(parts[0], 10);
					const numPhys = parseInt(parts[7], 10);
					const phys: number[] = [];
					for (let j = 0; j < numPhys; j++) phys.push(parseInt(parts[8 + j], 10));
					map.set(tag, phys);
				}
			};
			eat_entities(nCurves, 1);
			eat_entities(nSurfs, 2);
			eat_entities(nVols, 3);
			expect('$EndEntities');
		} else if (trimmed === '$Nodes') {
			// Header: numEntityBlocks numNodes minNodeTag maxNodeTag
			const hdr = (next() ?? '').split(/\s+/).map((s) => parseInt(s, 10));
			const numBlocks = hdr[0];
			for (let b = 0; b < numBlocks; b++) {
				// Block header: entityDim entityTag parametric numNodesInBlock
				const bh = (next() ?? '').split(/\s+/).map((s) => parseInt(s, 10));
				const numNodesInBlock = bh[3];
				const tags: number[] = [];
				for (let i = 0; i < numNodesInBlock; i++) tags.push(parseInt(next() ?? '', 10));
				for (let i = 0; i < numNodesInBlock; i++) {
					const parts = (next() ?? '').split(/\s+/);
					const x = parseFloat(parts[0]);
					const y = parseFloat(parts[1]);
					const z = parseFloat(parts[2]);
					node_idx_to_pos.set(tags[i], [x, y, z]);
				}
			}
			expect('$EndNodes');
		} else if (trimmed === '$Elements') {
			// Header: numEntityBlocks numElements minElementTag maxElementTag
			const hdr = (next() ?? '').split(/\s+/).map((s) => parseInt(s, 10));
			const numBlocks = hdr[0];
			for (let b = 0; b < numBlocks; b++) {
				// Block header: entityDim entityTag elementType numElementsInBlock
				const bh = (next() ?? '').split(/\s+/).map((s) => parseInt(s, 10));
				const entityDim = bh[0];
				const entityTag = bh[1];
				const elementType = bh[2];
				const numElems = bh[3];
				// elementType 2 = triangle (3 nodes), 4 = tet (4 nodes), 1 = line (skip), 15 = point (skip)
				const nodesPerElem = elementType === 2 ? 3 : elementType === 4 ? 4 : -1;
				const phys = entity_phys.get(entityDim)?.get(entityTag) ?? [];
				const phys_tag = phys[0] ?? 0;
				for (let i = 0; i < numElems; i++) {
					const parts = (next() ?? '').split(/\s+/).filter((s) => s.length);
					if (nodesPerElem < 0) continue;
					if (elementType === 2) {
						tris_raw.push(parseInt(parts[1], 10));
						tris_raw.push(parseInt(parts[2], 10));
						tris_raw.push(parseInt(parts[3], 10));
						tri_phys_raw.push(phys_tag);
					} else if (elementType === 4) {
						tets_raw.push(parseInt(parts[1], 10));
						tets_raw.push(parseInt(parts[2], 10));
						tets_raw.push(parseInt(parts[3], 10));
						tets_raw.push(parseInt(parts[4], 10));
						tet_phys_raw.push(phys_tag);
					}
				}
			}
			expect('$EndElements');
		}
		// Other sections (PartitionedEntities, Periodic, etc.) skipped.
	}

	// Build dense node array, remap gmsh tags → 0-indexed positions
	const node_count = node_idx_to_pos.size;
	const nodes = new Float64Array(node_count * 3);
	const remap = new Map<number, number>();
	let nidx = 0;
	const bbMin: [number, number, number] = [Infinity, Infinity, Infinity];
	const bbMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
	for (const [tag, pos] of node_idx_to_pos.entries()) {
		remap.set(tag, nidx);
		nodes[nidx * 3] = pos[0];
		nodes[nidx * 3 + 1] = pos[1];
		nodes[nidx * 3 + 2] = pos[2];
		for (let d = 0; d < 3; d++) {
			if (pos[d] < bbMin[d]) bbMin[d] = pos[d];
			if (pos[d] > bbMax[d]) bbMax[d] = pos[d];
		}
		nidx++;
	}

	const tris = new Uint32Array(tris_raw.length);
	for (let i = 0; i < tris_raw.length; i++) tris[i] = remap.get(tris_raw[i])!;
	const tets = new Uint32Array(tets_raw.length);
	for (let i = 0; i < tets_raw.length; i++) tets[i] = remap.get(tets_raw[i])!;

	return {
		nodes,
		tris,
		tri_phys: new Int32Array(tri_phys_raw),
		tets,
		tet_phys: new Int32Array(tet_phys_raw),
		phys_names,
		phys_dim,
		bbox: { min: bbMin, max: bbMax }
	};
}
