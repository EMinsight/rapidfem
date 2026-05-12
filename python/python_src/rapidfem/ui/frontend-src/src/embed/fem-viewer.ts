/**
 * <fem-viewer> — Embeddable rapidfem 3D viewer web component.
 *
 * Loads a baked example bundle (the same `<name>.json` + `.bin` artefacts
 * produced by `scripts/bake_demo.py`) and renders mesh + optional field
 * directly in any host page.
 *
 * Usage:
 *   <script src="https://fem.rapidpassives.org/embed/fem-viewer.js"></script>
 *   <fem-viewer src="/demo/wr90.json" rotate animate-field></fem-viewer>
 *
 * Attributes:
 *   src             URL to a baked `.json` (the matching `.bin` is auto-loaded)
 *   width / height  CSS dimensions (default 100% / 400px)
 *   rotate          continuous camera orbit
 *   interactive     enable mouse orbit/pan/zoom (default off)
 *   transparent     transparent background
 *   speed           animation speed multiplier (default 1)
 *   theta / phi     initial camera angles in degrees (default 45 / 45)
 *   show-geometry   render filled surface mesh (default on)
 *   show-mesh       render mesh edges (default off)
 *   show-field      render the point-cloud field viz (default on when present)
 *   field-mode      'lin' or 'log' (default 'lin')
 *   field-freq      frequency index for field display (default last)
 *   field-port      port index for field display (default 0)
 *   animate-field   oscillate the phase to show wave motion
 */

import {
	initGL, disposeGL, createCamera, addMesh, clearMeshes, setPointCloud,
	setPointPhase, setPointLinRange, setPointLogRange, setPointScaleMode,
	setBBox, render3D, fitCamera, type Camera, type GLState,
} from '../lib/render/canvas3d';

const FIELD_BIN_MAGIC = 0x52464d46; // "RFMF"

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
	name: string;
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
	if (dv.getUint32(0, true) !== FIELD_BIN_MAGIC) {
		throw new Error('field bin: bad magic');
	}
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

// Extract the outer hull of every physical volume by finding tet faces that
// appear only once (a face shared by two tets is internal). Same algorithm
// MeshViewer uses to render the substrate / air / PML shells. This is the
// dominant visible surface — the named-surface tris (port faces, PEC walls)
// alone would only show "slices" of the geometry.
function buildVolumeBoundaryTris(mesh: MeshPayload): number[] {
	const tets = mesh.tets;
	const tet_phys = mesh.tet_phys;
	const ntets = tet_phys.length;
	const enc = (a: number, b: number, c: number): bigint => {
		const s = [a, b, c].sort((x, y) => x - y);
		return (BigInt(s[0]) * 0x100000000n + BigInt(s[1])) * 0x100000000n + BigInt(s[2]);
	};
	// Group tets by phys-volume tag.
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

// Build a flat-shaded triangle-soup (positions + normals) from a list of
// triangle indices into the mesh's flat node array.
function buildTriSoup(
	nodes: number[],
	tris: number[],
): { positions: Float32Array; normals: Float32Array } {
	const n_tris = tris.length / 3;
	const positions = new Float32Array(n_tris * 9);
	const normals = new Float32Array(n_tris * 9);
	const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0], ab = [0, 0, 0], ac = [0, 0, 0];
	for (let t = 0; t < n_tris; t++) {
		const i = tris[t * 3], j = tris[t * 3 + 1], k = tris[t * 3 + 2];
		a[0] = nodes[i * 3]; a[1] = nodes[i * 3 + 1]; a[2] = nodes[i * 3 + 2];
		b[0] = nodes[j * 3]; b[1] = nodes[j * 3 + 1]; b[2] = nodes[j * 3 + 2];
		c[0] = nodes[k * 3]; c[1] = nodes[k * 3 + 1]; c[2] = nodes[k * 3 + 2];
		ab[0] = b[0] - a[0]; ab[1] = b[1] - a[1]; ab[2] = b[2] - a[2];
		ac[0] = c[0] - a[0]; ac[1] = c[1] - a[1]; ac[2] = c[2] - a[2];
		// Cross product → flat normal
		const nx = ab[1] * ac[2] - ab[2] * ac[1];
		const ny = ab[2] * ac[0] - ab[0] * ac[2];
		const nz = ab[0] * ac[1] - ab[1] * ac[0];
		const inv = 1 / Math.max(Math.hypot(nx, ny, nz), 1e-20);
		const nxN = nx * inv, nyN = ny * inv, nzN = nz * inv;
		positions.set(a, t * 9); positions.set(b, t * 9 + 3); positions.set(c, t * 9 + 6);
		normals[t * 9] = nxN; normals[t * 9 + 1] = nyN; normals[t * 9 + 2] = nzN;
		normals[t * 9 + 3] = nxN; normals[t * 9 + 4] = nyN; normals[t * 9 + 5] = nzN;
		normals[t * 9 + 6] = nxN; normals[t * 9 + 7] = nyN; normals[t * 9 + 8] = nzN;
	}
	return { positions, normals };
}

// Sample a point cloud from tet centroids with the per-node field interpolated
// (linear interp from the 4 tet vertices to centroid).
function buildFieldPointCloud(
	mesh: MeshPayload,
	field: number[],   // flat [A,B,C,A,B,C,...] per node
): { positions: Float32Array; abc: Float32Array } {
	const nodes = mesh.nodes;
	const tets = mesh.tets;
	const n_tets = tets.length / 4;
	const positions = new Float32Array(n_tets * 3);
	const abc = new Float32Array(n_tets * 3);
	for (let t = 0; t < n_tets; t++) {
		const i0 = tets[t * 4], i1 = tets[t * 4 + 1],
		      i2 = tets[t * 4 + 2], i3 = tets[t * 4 + 3];
		positions[t * 3]     = 0.25 * (nodes[i0 * 3]     + nodes[i1 * 3]     + nodes[i2 * 3]     + nodes[i3 * 3]);
		positions[t * 3 + 1] = 0.25 * (nodes[i0 * 3 + 1] + nodes[i1 * 3 + 1] + nodes[i2 * 3 + 1] + nodes[i3 * 3 + 1]);
		positions[t * 3 + 2] = 0.25 * (nodes[i0 * 3 + 2] + nodes[i1 * 3 + 2] + nodes[i2 * 3 + 2] + nodes[i3 * 3 + 2]);
		abc[t * 3]     = 0.25 * (field[i0 * 3]     + field[i1 * 3]     + field[i2 * 3]     + field[i3 * 3]);
		abc[t * 3 + 1] = 0.25 * (field[i0 * 3 + 1] + field[i1 * 3 + 1] + field[i2 * 3 + 1] + field[i3 * 3 + 1]);
		abc[t * 3 + 2] = 0.25 * (field[i0 * 3 + 2] + field[i1 * 3 + 2] + field[i2 * 3 + 2] + field[i3 * 3 + 2]);
	}
	return { positions, abc };
}

// Compute a sensible {floor, range} for the field magnitude colour mapping.
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
	private glState: GLState | null = null;
	private camera: Camera = createCamera();
	private animId = 0;
	private mounted = false;
	private mesh: MeshPayload | null = null;
	private fields: (number[] | null)[][] | null = null;
	private needsRender = false;
	private isDragging = false;
	private isRightDrag = false;
	private lastMouse = { x: 0, y: 0 };

	static get observedAttributes() {
		return ['src', 'width', 'height', 'rotate', 'interactive', 'transparent', 'speed',
		        'theta', 'phi', 'show-geometry', 'show-mesh', 'show-field',
		        'field-mode', 'field-freq', 'field-port', 'animate-field'];
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
		this.loadingEl.textContent = '';
		this.wrapper.appendChild(this.loadingEl);
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
		else if (name === 'field-mode' || name === 'field-freq' || name === 'field-port') {
			this.applyField();
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
		if (this.loadingEl) { this.loadingEl.textContent = 'Loading...'; this.loadingEl.style.display = 'flex'; }
		try {
			const url = new URL(srcUrl, location.href).href;
			const resp = await fetch(url);
			if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
			const baked: BakedExample = await resp.json();

			this.mesh = extractLatestMesh(baked);
			if (!this.mesh) throw new Error('no mesh payload in bake');
			const result = extractLatestResult(baked);
			if (result && result.fields && (result.fields as BakedFieldsStub).$bin) {
				if (this.loadingEl) this.loadingEl.textContent = 'Decoding field...';
				this.fields = await hydrateFields(result.fields as BakedFieldsStub, url);
			} else {
				this.fields = null;
			}

			this.rebuildScene();
			this.fitView();
			if (this.loadingEl) this.loadingEl.style.display = 'none';
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

		const showGeom = this.attrBool('show-geometry', true);
		if (showGeom) {
			// Per-volume outer hull (substrate / air / PML shells). This is
			// the main "shape" of the model. Slightly translucent so the
			// field point cloud and inner ports remain visible.
			const hull_tris = buildVolumeBoundaryTris(this.mesh);
			if (hull_tris.length) {
				const { positions, normals } = buildTriSoup(this.mesh.nodes, hull_tris);
				addMesh(this.glState, positions, normals, [0.16, 0.16, 0.20], 1);
			}
			// Named-surface tris (PEC walls, ports). Brighter so they stand
			// out against the hull.
			if (this.mesh.tris.length) {
				const { positions, normals } = buildTriSoup(this.mesh.nodes, this.mesh.tris);
				addMesh(this.glState, positions, normals, [0.36, 0.30, 0.34], 2);
			}
		}
		this.applyField();
	}

	private applyField() {
		if (!this.glState || !this.mesh) return;
		const showField = this.attrBool('show-field', this.fields !== null);
		if (!showField || !this.fields) { setPointCloud(this.glState, new Float32Array(0), new Float32Array(0)); return; }
		const fi = Math.min(parseInt(this.getAttribute('field-freq') || '-1', 10), this.fields.length - 1);
		const fIdx = fi >= 0 ? fi : this.fields.length - 1;
		const pIdx = parseInt(this.getAttribute('field-port') || '0', 10);
		const row = this.fields[fIdx];
		const arr = row && row[pIdx];
		if (!arr) { setPointCloud(this.glState, new Float32Array(0), new Float32Array(0)); return; }
		const { positions, abc } = buildFieldPointCloud(this.mesh, arr);
		setPointCloud(this.glState, positions, abc);
		const mode = (this.getAttribute('field-mode') || 'lin') === 'log' ? 'log' as const : 'lin' as const;
		const r = computeFieldRange(abc, mode);
		setPointScaleMode(this.glState, mode);
		if (mode === 'log') setPointLogRange(this.glState, r.floor, r.range);
		else                setPointLinRange(this.glState, r.floor, r.range);
	}

	private attrBool(name: string, def: boolean): boolean {
		if (!this.hasAttribute(name)) return def;
		const v = this.getAttribute(name);
		return v === null || v === '' || v === 'true' || v === '1';
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

	private renderFrame(time = 0) {
		if (!this.glState || !this.canvas || !this.mounted) return;
		const { w, h } = this.syncCanvas();
		if (w <= 0 || h <= 0) return;
		const doRotate = this.hasAttribute('rotate');
		const animateField = this.hasAttribute('animate-field');
		const isTransparent = this.hasAttribute('transparent');
		const speed = parseFloat(this.getAttribute('speed') || '1');
		if (doRotate && !this.isDragging) {
			this.camera = { ...this.camera, theta: this.camera.theta + 0.003 * speed };
		}
		if (animateField) {
			const phase = (time * 0.001 * speed) % (2 * Math.PI);
			setPointPhase(this.glState, phase);
		}
		render3D(this.glState, this.camera, w, h, 0, isTransparent, null, 1.0, null);
		this.needsRender = false;
	}

	private startAnimation() {
		const id = ++this.animId;
		const isAnimated = this.hasAttribute('rotate') || this.hasAttribute('animate-field');
		if (!isAnimated && !this.hasAttribute('interactive')) {
			this.renderFrame(0);
			return;
		}
		const tick = (time: number) => {
			if (!this.mounted || id !== this.animId) return;
			if (isAnimated || this.needsRender) this.renderFrame(time);
			requestAnimationFrame(tick);
		};
		requestAnimationFrame(tick);
	}
}

if (typeof customElements !== 'undefined' && !customElements.get('fem-viewer')) {
	customElements.define('fem-viewer', FemViewerElement);
}
