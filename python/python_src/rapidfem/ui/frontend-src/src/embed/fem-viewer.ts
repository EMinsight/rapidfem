/**
 * <fem-viewer> — Embeddable RapidFEM 3D viewer web component.
 *
 * Loads a baked example bundle (the `<name>.json` + `.bin` artefacts
 * produced by `scripts/bake_demo.py`) and renders it in any host page.
 *
 * Usage:
 *   <script src="https://fem.rapidpassives.org/embed/fem-viewer.js"></script>
 *   <fem-viewer src="/demo/wr90.json" rotate cycle></fem-viewer>
 *
 * Attributes:
 *   src             URL to a baked `.json` (the matching `.bin` is auto-loaded)
 *   width / height  CSS dimensions (default 100% / 400px)
 *   rotate          continuous camera orbit
 *   cycle           cycle the display through Geometry → Mesh → Field
 *   mode            static display mode: `geometry` | `mesh` | `field`
 *                   (overridden by `cycle` if both are set; default `geometry`)
 *   interactive     enable mouse orbit/pan/zoom (default off)
 *   transparent     transparent background
 *   speed           animation speed multiplier (default 1)
 *   theta / phi     initial camera angles in degrees (default 45 / 45)
 *   field-mode      'lin' or 'log' (default 'lin')
 *   field-freq      frequency index for field display (default last)
 *   field-port      port index for field display (default 0)
 */

import {
	initGL, disposeGL, createCamera, addMesh, addLineMesh, clearMeshes,
	setPointCloud, setPointLinRange, setPointLogRange, setPointScaleMode,
	setTagVisible, setBBox, render3D, fitCamera,
	type Camera, type GLState,
} from '../lib/render/canvas3d';

const FIELD_BIN_MAGIC = 0x52464d46; // "RFMF"

// Tag layout — each renderable group gets its own integer so we can
// toggle visibility per cycle phase via setTagVisible(state, tag, vis).
const TAG_HULL = 1;        // dielectric / volume hulls (substrate, air, PML)
const TAG_CONDUCTORS = 2;  // named PEC walls + ports (mesh.tris)
const TAG_WIRE = 3;        // wireframe edges

// Cycle phases in display order; each holds for CYCLE_HOLD_S seconds.
type Phase = 'geometry' | 'mesh' | 'field';
const CYCLE_ORDER: Phase[] = ['geometry', 'mesh', 'field'];
const CYCLE_HOLD_S = 2.4;

interface BakedFieldsStub {
	$bin: true; magic: number; version: number;
	n_freq: number; n_port: number; stride: number; url: string;
}
interface MeshPayload {
	bbox: { min: [number, number, number]; max: [number, number, number] };
	nodes: number[];
	tris: number[];
	tri_phys: number[];
	tets: number[];
	tet_phys: number[];
	phys_names: Record<string, string>;
	stats: { n_nodes: number; n_tets: number };
}
interface ResultPayload {
	frequencies: number[];
	sparams: number[][][][];
	n_freq: number;
	fields?: BakedFieldsStub | (number[] | null)[][];
}
interface BakedExample {
	cells: Array<{
		display_events: Array<{
			kind: 'geometry' | 'mesh' | 'result' | 'error';
			payload?: MeshPayload | ResultPayload | Record<string, unknown>;
		}>;
	}>;
}

function extractLatestMesh(baked: BakedExample): MeshPayload | null {
	for (let i = baked.cells.length - 1; i >= 0; i--) {
		for (const ev of baked.cells[i].display_events) {
			if (ev.kind === 'mesh' && ev.payload) return ev.payload as MeshPayload;
		}
	}
	return null;
}
function extractLatestResult(baked: BakedExample): ResultPayload | null {
	for (let i = baked.cells.length - 1; i >= 0; i--) {
		for (const ev of baked.cells[i].display_events) {
			if (ev.kind === 'result' && ev.payload) return ev.payload as ResultPayload;
		}
	}
	return null;
}

async function hydrateFields(stub: BakedFieldsStub, baseUrl: string): Promise<(number[] | null)[][]> {
	const url = new URL(stub.url, baseUrl).href;
	const resp = await fetch(url);
	if (!resp.ok) throw new Error(`bin fetch ${resp.status}`);
	const buf = await resp.arrayBuffer();
	const dv = new DataView(buf);
	if (dv.getUint32(0, true) !== FIELD_BIN_MAGIC) throw new Error('field bin: bad magic');
	const n_freq = dv.getUint32(8, true);
	const n_port = dv.getUint32(12, true);
	const stride = dv.getUint32(16, true);
	const mask_off = 20;
	const mask = new Uint8Array(buf, mask_off, n_freq * n_port);
	const floats_off = mask_off + ((mask.byteLength + 3) & ~3);
	const all = new Float32Array(buf, floats_off);
	const out: (number[] | null)[][] = [];
	let cursor = 0;
	for (let f = 0; f < n_freq; f++) {
		const row: (number[] | null)[] = [];
		for (let p = 0; p < n_port; p++) {
			if (mask[f * n_port + p] === 0) row.push(null);
			else { row.push(Array.from(all.subarray(cursor, cursor + stride))); cursor += stride; }
		}
		out.push(row);
	}
	return out;
}

// Flat-shaded triangle soup from indexed mesh nodes.
function buildTriSoup(
	nodes: number[],
	tris: number[],
): { positions: Float32Array; normals: Float32Array } {
	const n_tris = tris.length / 3;
	const positions = new Float32Array(n_tris * 9);
	const normals = new Float32Array(n_tris * 9);
	const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0];
	for (let t = 0; t < n_tris; t++) {
		const i = tris[t * 3], j = tris[t * 3 + 1], k = tris[t * 3 + 2];
		a[0] = nodes[i * 3]; a[1] = nodes[i * 3 + 1]; a[2] = nodes[i * 3 + 2];
		b[0] = nodes[j * 3]; b[1] = nodes[j * 3 + 1]; b[2] = nodes[j * 3 + 2];
		c[0] = nodes[k * 3]; c[1] = nodes[k * 3 + 1]; c[2] = nodes[k * 3 + 2];
		const e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
		const e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
		const nx = e1y * e2z - e1z * e2y;
		const ny = e1z * e2x - e1x * e2z;
		const nz = e1x * e2y - e1y * e2x;
		const inv = 1 / Math.max(Math.hypot(nx, ny, nz), 1e-20);
		const nxN = nx * inv, nyN = ny * inv, nzN = nz * inv;
		positions.set(a, t * 9); positions.set(b, t * 9 + 3); positions.set(c, t * 9 + 6);
		for (let v = 0; v < 3; v++) {
			normals[t * 9 + v * 3] = nxN;
			normals[t * 9 + v * 3 + 1] = nyN;
			normals[t * 9 + v * 3 + 2] = nzN;
		}
	}
	return { positions, normals };
}

// Per-volume outer hull from tet boundary faces (the substrate / air /
// PML shells that bound each physical-group volume).
function buildVolumeHullTris(mesh: MeshPayload): number[] {
	const tets = mesh.tets;
	const tet_phys = mesh.tet_phys;
	const ntets = tet_phys.length;
	const enc = (a: number, b: number, c: number): bigint => {
		const s = [a, b, c].sort((x, y) => x - y);
		return (BigInt(s[0]) * 0x100000000n + BigInt(s[1])) * 0x100000000n + BigInt(s[2]);
	};
	const per_vol = new Map<number, number[]>();
	for (let t = 0; t < ntets; t++) {
		const v = tet_phys[t];
		if (!v) continue;
		let arr = per_vol.get(v);
		if (!arr) { arr = []; per_vol.set(v, arr); }
		arr.push(t);
	}
	const out: number[] = [];
	for (const [, tet_indices] of per_vol.entries()) {
		const seen = new Map<bigint, { count: number; tri: [number, number, number] }>();
		for (const t of tet_indices) {
			const a = tets[t * 4], b = tets[t * 4 + 1],
			      c = tets[t * 4 + 2], d = tets[t * 4 + 3];
			const faces: [number, number, number][] = [
				[a, b, c], [a, b, d], [a, c, d], [b, c, d],
			];
			for (const f of faces) {
				const k = enc(f[0], f[1], f[2]);
				const prev = seen.get(k);
				if (!prev) seen.set(k, { count: 1, tri: f });
				else prev.count++;
			}
		}
		for (const e of seen.values()) {
			if (e.count === 1) out.push(e.tri[0], e.tri[1], e.tri[2]);
		}
	}
	return out;
}

// Build edge-list (positions, two vertices per edge) from triangle faces,
// deduplicated. Used for the wireframe / mesh-mode display.
function buildEdgeLines(nodes: number[], tris: number[]): Float32Array {
	const seen = new Set<bigint>();
	const out: number[] = [];
	const push = (a: number, b: number) => {
		const lo = a < b ? a : b, hi = a < b ? b : a;
		const k = (BigInt(lo) << 32n) | BigInt(hi);
		if (seen.has(k)) return;
		seen.add(k);
		out.push(
			nodes[a * 3], nodes[a * 3 + 1], nodes[a * 3 + 2],
			nodes[b * 3], nodes[b * 3 + 1], nodes[b * 3 + 2],
		);
	};
	const n_tris = tris.length / 3;
	for (let t = 0; t < n_tris; t++) {
		const i = tris[t * 3], j = tris[t * 3 + 1], k = tris[t * 3 + 2];
		push(i, j); push(j, k); push(k, i);
	}
	return Float32Array.from(out);
}

// Tet-centroid point cloud for the field viz.
function buildFieldCloud(
	mesh: MeshPayload,
	field: number[],   // flat [A,B,C, A,B,C, ...] per node
): { positions: Float32Array; abc: Float32Array } {
	const nodes = mesh.nodes;
	const tets = mesh.tets;
	const n_tets = tets.length / 4;
	const positions = new Float32Array(n_tets * 3);
	const abc = new Float32Array(n_tets * 3);
	for (let t = 0; t < n_tets; t++) {
		const i0 = tets[t * 4], i1 = tets[t * 4 + 1],
		      i2 = tets[t * 4 + 2], i3 = tets[t * 4 + 3];
		for (let d = 0; d < 3; d++) {
			positions[t * 3 + d] = 0.25 * (
				nodes[i0 * 3 + d] + nodes[i1 * 3 + d] +
				nodes[i2 * 3 + d] + nodes[i3 * 3 + d]
			);
			abc[t * 3 + d] = 0.25 * (
				field[i0 * 3 + d] + field[i1 * 3 + d] +
				field[i2 * 3 + d] + field[i3 * 3 + d]
			);
		}
	}
	return { positions, abc };
}

function computeFieldRange(abc: Float32Array, mode: 'lin' | 'log'): { floor: number; range: number } {
	let max_e2 = 0;
	for (let i = 0; i < abc.length; i += 3) {
		const e2 = Math.max(abc[i], abc[i + 1]);
		if (e2 > max_e2) max_e2 = e2;
	}
	const max_e = Math.sqrt(Math.max(max_e2, 1e-30));
	if (mode === 'log') {
		const log_max = Math.log10(max_e);
		return { floor: log_max - 4, range: 4 };
	}
	return { floor: 0, range: max_e };
}

class FemViewerElement extends HTMLElement {
	private canvas: HTMLCanvasElement | null = null;
	private wrapper: HTMLDivElement | null = null;
	private loadingEl: HTMLDivElement | null = null;
	private labelEl: HTMLDivElement | null = null;
	private glState: GLState | null = null;
	private camera: Camera = createCamera();
	private animId = 0;
	private mounted = false;
	private mesh: MeshPayload | null = null;
	private fields: (number[] | null)[][] | null = null;
	private hasField = false;
	private needsRender = false;
	private isDragging = false;
	private isRightDrag = false;
	private lastMouse = { x: 0, y: 0 };
	private currentPhase: Phase = 'geometry';
	private phaseStart = 0;

	static get observedAttributes() {
		return ['src', 'width', 'height', 'rotate', 'cycle', 'mode', 'interactive',
		        'transparent', 'speed', 'theta', 'phi',
		        'field-mode', 'field-freq', 'field-port'];
	}

	connectedCallback() {
		this.mounted = true;
		const shadow = this.attachShadow({ mode: 'open' });
		const isTransparent = this.hasAttribute('transparent');
		this.wrapper = document.createElement('div');
		this.wrapper.style.cssText = `position:relative;width:${this.getAttribute('width') || '100%'};height:${this.getAttribute('height') || '400px'};background:${isTransparent ? 'transparent' : '#131316'};overflow:hidden;border-radius:inherit;`;
		this.canvas = document.createElement('canvas');
		this.canvas.style.cssText = `display:block;width:100%;height:100%;cursor:${this.hasAttribute('interactive') ? 'grab' : 'default'};`;
		this.wrapper.appendChild(this.canvas);

		this.loadingEl = document.createElement('div');
		this.loadingEl.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font:500 11px/1 monospace;color:#55535a;';
		this.wrapper.appendChild(this.loadingEl);

		// Phase label (visible only when cycling)
		this.labelEl = document.createElement('div');
		this.labelEl.style.cssText = 'position:absolute;top:8px;left:10px;font:500 10px/1 monospace;color:#9a96a0;text-transform:uppercase;letter-spacing:0.5px;pointer-events:none;opacity:0;transition:opacity 0.3s;';
		this.wrapper.appendChild(this.labelEl);

		const badge = document.createElement('a');
		badge.href = 'https://fem.rapidpassives.org';
		badge.target = '_blank'; badge.rel = 'noopener';
		badge.textContent = 'RapidFEM';
		badge.style.cssText = 'position:absolute;bottom:6px;right:8px;font:500 9px/1 monospace;color:#55535a;text-decoration:none;opacity:0.7;transition:opacity 0.15s;';
		badge.onmouseenter = () => badge.style.opacity = '1';
		badge.onmouseleave = () => badge.style.opacity = '0.7';
		this.wrapper.appendChild(badge);

		shadow.appendChild(this.wrapper);

		this.glState = initGL(this.canvas);
		if (!this.glState) return;

		if (this.hasAttribute('interactive')) {
			this.canvas.addEventListener('pointerdown', (e) => this.onPointerDown(e));
			this.canvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
			this.canvas.addEventListener('pointerup', () => this.onPointerUp());
			this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
			this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
			this.canvas.addEventListener('dblclick', () => this.fitView());
		}
		const ro = new ResizeObserver(() => { this.needsRender = true; });
		ro.observe(this.wrapper);

		const src = this.getAttribute('src');
		if (src) void this.load(src);
	}

	disconnectedCallback() {
		this.mounted = false; this.animId++;
		if (this.glState) disposeGL(this.glState);
	}

	attributeChangedCallback(name: string, _old: string | null, val: string | null) {
		if (!this.mounted) return;
		if (name === 'src' && val) void this.load(val);
		else if (name === 'mode' || name === 'field-mode' || name === 'field-freq' || name === 'field-port') {
			this.applyField();
			this.applyPhase(this.resolvePhase());
			this.needsRender = true;
		}
	}

	private onPointerDown(e: PointerEvent) {
		this.isDragging = true; this.isRightDrag = e.button === 2;
		this.lastMouse = { x: e.clientX, y: e.clientY };
		this.canvas?.setPointerCapture(e.pointerId);
		if (this.canvas) this.canvas.style.cursor = 'grabbing';
	}
	private onPointerMove(e: PointerEvent) {
		if (!this.isDragging) return;
		const dx = e.clientX - this.lastMouse.x;
		const dy = e.clientY - this.lastMouse.y;
		this.lastMouse = { x: e.clientX, y: e.clientY };
		if (this.isRightDrag) {
			const s = this.camera.distance * 0.0007;
			const ct = Math.cos(this.camera.theta), st = Math.sin(this.camera.theta);
			this.camera = {
				...this.camera,
				target: [
					this.camera.target[0] + (dx * ct - dy * st * Math.sin(this.camera.phi)) * s,
					this.camera.target[1] - (dx * st + dy * ct * Math.sin(this.camera.phi)) * s,
					this.camera.target[2] + dy * Math.cos(this.camera.phi) * s,
				],
			};
		} else {
			this.camera = {
				...this.camera,
				theta: this.camera.theta + dx * 0.005,
				phi: Math.max(0.05, Math.min(Math.PI / 2 - 0.05, this.camera.phi + dy * 0.005)),
			};
		}
		this.needsRender = true;
	}
	private onPointerUp() {
		this.isDragging = false; this.isRightDrag = false;
		if (this.canvas) this.canvas.style.cursor = 'grab';
	}
	private onWheel(e: WheelEvent) {
		e.preventDefault();
		this.camera = { ...this.camera, distance: this.camera.distance * (e.deltaY > 0 ? 1.1 : 1 / 1.1) };
		this.needsRender = true;
	}

	private fitView() {
		if (!this.glState || !this.mesh) return;
		this.camera = fitCamera(this.mesh.bbox.min, this.mesh.bbox.max);
		const theta = parseFloat(this.getAttribute('theta') || '45') * Math.PI / 180;
		const phi = parseFloat(this.getAttribute('phi') || '45') * Math.PI / 180;
		this.camera = { ...this.camera, theta, phi };
		this.needsRender = true;
	}

	private async load(srcUrl: string) {
		if (this.loadingEl) { this.loadingEl.textContent = 'Loading…'; this.loadingEl.style.display = 'flex'; }
		try {
			const url = new URL(srcUrl, location.href).href;
			const resp = await fetch(url);
			if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
			const baked: BakedExample = await resp.json();

			this.mesh = extractLatestMesh(baked);
			if (!this.mesh) throw new Error('no mesh payload in bake');
			const result = extractLatestResult(baked);
			if (result && result.fields && (result.fields as BakedFieldsStub).$bin) {
				if (this.loadingEl) this.loadingEl.textContent = 'Decoding field…';
				this.fields = await hydrateFields(result.fields as BakedFieldsStub, url);
				this.hasField = this.fields.some((row) => row.some((x) => x !== null));
			} else {
				this.fields = null;
				this.hasField = false;
			}

			this.rebuildScene();
			this.fitView();
			if (this.loadingEl) this.loadingEl.style.display = 'none';
			this.phaseStart = performance.now() / 1000;
			this.applyPhase(this.resolvePhase());
			this.startAnimation();
		} catch (e) {
			console.error('[fem-viewer] load failed', e);
			if (this.loadingEl) this.loadingEl.textContent = `Error: ${(e as Error).message}`;
		}
	}

	private rebuildScene() {
		if (!this.glState || !this.mesh) return;
		clearMeshes(this.glState);
		setBBox(this.glState, this.mesh.bbox.min, this.mesh.bbox.max);

		// 1) Volume hulls — substrate / air / PML shells.
		const hull = buildVolumeHullTris(this.mesh);
		if (hull.length) {
			const { positions, normals } = buildTriSoup(this.mesh.nodes, hull);
			addMesh(this.glState, positions, normals, [0.20, 0.20, 0.24], TAG_HULL);
		}
		// 2) Named conductors / ports / PEC walls (mesh.tris).
		if (this.mesh.tris.length) {
			const { positions, normals } = buildTriSoup(this.mesh.nodes, this.mesh.tris);
			addMesh(this.glState, positions, normals, [0.55, 0.42, 0.45], TAG_CONDUCTORS);
		}
		// 3) Wireframe — edges of every surface tri, hidden by default; the
		//    mesh-mode phase toggles it on.
		const edges = buildEdgeLines(this.mesh.nodes, this.mesh.tris);
		if (edges.length) {
			addLineMesh(this.glState, edges, [0.32, 0.32, 0.38], TAG_WIRE);
		}
		this.applyField();
	}

	private applyField() {
		if (!this.glState || !this.mesh) return;
		if (!this.hasField || !this.fields) {
			setPointCloud(this.glState, new Float32Array(0), new Float32Array(0));
			return;
		}
		const fi = Math.min(parseInt(this.getAttribute('field-freq') || '-1', 10), this.fields.length - 1);
		const fIdx = fi >= 0 ? fi : this.fields.length - 1;
		const pIdx = parseInt(this.getAttribute('field-port') || '0', 10);
		const row = this.fields[fIdx];
		const arr = row && row[pIdx];
		if (!arr) { setPointCloud(this.glState, new Float32Array(0), new Float32Array(0)); return; }
		const { positions, abc } = buildFieldCloud(this.mesh, arr);
		setPointCloud(this.glState, positions, abc);
		const mode = (this.getAttribute('field-mode') || 'lin') === 'log' ? 'log' as const : 'lin' as const;
		const r = computeFieldRange(abc, mode);
		setPointScaleMode(this.glState, mode);
		if (mode === 'log') setPointLogRange(this.glState, r.floor, r.range);
		else                setPointLinRange(this.glState, r.floor, r.range);
	}

	/** What phase should be displayed right now. */
	private resolvePhase(): Phase {
		if (this.hasAttribute('cycle')) {
			// Time-based phase cycling.
			const t = performance.now() / 1000 - this.phaseStart;
			let order: Phase[] = this.hasField ? CYCLE_ORDER : CYCLE_ORDER.filter(p => p !== 'field');
			const idx = Math.floor(t / CYCLE_HOLD_S) % order.length;
			return order[idx];
		}
		const mode = (this.getAttribute('mode') || 'geometry').toLowerCase();
		if (mode === 'mesh' || mode === 'field') return mode as Phase;
		return 'geometry';
	}

	/** Toggle which mesh groups + field cloud are visible for this phase. */
	private applyPhase(phase: Phase) {
		if (!this.glState) return;
		this.currentPhase = phase;
		const showHull = phase === 'geometry' || phase === 'field';
		const showCond = phase === 'geometry' || phase === 'field';
		const showWire = phase === 'mesh';
		const showField = phase === 'field' && this.hasField;
		setTagVisible(this.glState, TAG_HULL, showHull);
		setTagVisible(this.glState, TAG_CONDUCTORS, showCond);
		setTagVisible(this.glState, TAG_WIRE, showWire);
		// Point cloud: not tag-gated; clear it when the phase isn't field.
		if (!showField && this.mesh) {
			setPointCloud(this.glState, new Float32Array(0), new Float32Array(0));
		} else if (showField) {
			this.applyField();
		}
		if (this.labelEl) {
			this.labelEl.textContent = phase;
			this.labelEl.style.opacity = this.hasAttribute('cycle') ? '0.65' : '0';
		}
	}

	private syncCanvas(): { w: number; h: number } {
		if (!this.canvas) return { w: 0, h: 0 };
		const rect = this.canvas.getBoundingClientRect();
		const w = Math.round(rect.width), h = Math.round(rect.height);
		if (w <= 0 || h <= 0) return { w, h };
		const dpr = window.devicePixelRatio || 1;
		const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
		if (this.canvas.width !== bw || this.canvas.height !== bh) {
			this.canvas.width = bw; this.canvas.height = bh;
		}
		return { w, h };
	}

	private renderFrame() {
		if (!this.glState || !this.canvas || !this.mounted) return;
		const { w, h } = this.syncCanvas();
		if (w <= 0 || h <= 0) return;
		const isTransparent = this.hasAttribute('transparent');
		const speed = parseFloat(this.getAttribute('speed') || '1');
		if (this.hasAttribute('rotate') && !this.isDragging) {
			this.camera = { ...this.camera, theta: this.camera.theta + 0.003 * speed };
		}
		if (this.hasAttribute('cycle')) {
			const next = this.resolvePhase();
			if (next !== this.currentPhase) this.applyPhase(next);
		}
		// 5th arg is zFlip — MUST be 1 in the rapidfem renderer, otherwise the
		// vertex shader multiplies every vertex's z by 0 and the geometry
		// collapses to a horizontal slice at z=0. (Lines stay correct because
		// the line shader doesn't reference uZFlip.)
		render3D(this.glState, this.camera, w, h, 1);
		this.needsRender = false;
	}

	private startAnimation() {
		const id = ++this.animId;
		const animated = this.hasAttribute('rotate') || this.hasAttribute('cycle');
		if (!animated && !this.hasAttribute('interactive')) {
			this.renderFrame();
			return;
		}
		const tick = () => {
			if (!this.mounted || id !== this.animId) return;
			if (animated || this.needsRender) this.renderFrame();
			requestAnimationFrame(tick);
		};
		requestAnimationFrame(tick);
	}
}

if (typeof customElements !== 'undefined' && !customElements.get('fem-viewer')) {
	customElements.define('fem-viewer', FemViewerElement);
}
