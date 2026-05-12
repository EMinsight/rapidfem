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
 *   field-samples   N random points sampled in the volume (default 8000;
 *                   bump for full-page embeds, drop for tiny thumbnails)
 */

import {
	initGL, disposeGL, createCamera, clearMeshes,
	setPointCloud, setPointLinRange, setPointLogRange, setPointScaleMode,
	render3D, fitCamera,
	type Camera, type GLState,
} from '../lib/render/canvas3d';
import { buildScene, clearFieldCloud, sampleFieldCloud } from '../lib/render/scene_builder';

const FIELD_BIN_MAGIC = 0x52464d46; // "RFMF"

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

// (per-volume hull + edge extraction live in lib/render/mesh_scene.ts,
// shared with the in-app MeshViewer to keep the two pipelines bit-identical)

// (field sampling lives in scene_builder.ts — same algorithm the in-app
// viewer's viz.ts worker uses; we just inline the call here)

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
		        'field-mode', 'field-freq', 'field-port', 'field-samples'];
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
		else if (name === 'mode' || name === 'field-mode' || name === 'field-freq' || name === 'field-port' || name === 'field-samples') {
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

	/** Wipe + rebuild the GL scene for the current phase.
	 *  Routes through the same `buildScene` the in-app MeshViewer uses
	 *  → bit-identical rendering, no embed-specific hand-rolled colors. */
	private rebuildScene() {
		if (!this.glState || !this.mesh) return;
		clearMeshes(this.glState);
		const phase = this.currentPhase;
		// Field mode = pure point cloud; the in-app viewer keeps faces in
		// field mode and colors them by per-node |E|, but the embed only
		// ships per-tet centroid samples in the bake bundle, so we drop
		// the geometry entirely to avoid mixing the two.
		buildScene(this.glState, this.mesh, {
			showFaces: phase === 'geometry',
			showWire: phase === 'mesh',
		});
		if (phase === 'field' && this.hasField) this.applyField();
		else clearFieldCloud(this.glState);
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
		// Volume-uniform barycentric sampling — same algorithm as the
		// in-app worker (lib/viz.ts), just sync inline. Default 8 k keeps
		// the smaller embed canvas responsive; override via the
		// `field-samples` attribute (e.g. larger for a full-page embed).
		const n = Math.max(500, parseInt(this.getAttribute('field-samples') || '8000', 10));
		const { positions, abc, maxE2, minE2 } = sampleFieldCloud(this.mesh, arr, n);
		setPointCloud(this.glState, positions, abc);
		const mode = (this.getAttribute('field-mode') || 'lin') === 'log' ? 'log' as const : 'lin' as const;
		setPointScaleMode(this.glState, mode);
		if (mode === 'log') {
			const log_max = Math.log10(Math.sqrt(Math.max(maxE2, 1e-30)));
			setPointLogRange(this.glState, log_max - 4, 4);
		} else {
			setPointLinRange(this.glState, 0, Math.sqrt(Math.max(maxE2, 1e-30)));
		}
		void minE2;
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

	/** Apply a phase: re-runs buildScene with the appropriate flags so the
	 *  GL state holds exactly what this phase wants — no leftover tags
	 *  from the previous phase, no setTagVisible juggling. */
	private applyPhase(phase: Phase) {
		if (!this.glState) return;
		this.currentPhase = phase;
		this.rebuildScene();
		if (this.labelEl) {
			this.labelEl.textContent = phase;
			this.labelEl.style.opacity = this.hasAttribute('cycle') ? '0.65' : '0';
		}
	}

	private syncCanvas(): { w: number; h: number } {
		if (!this.canvas) return { w: 0, h: 0 };
		const rect = this.canvas.getBoundingClientRect();
		const cssW = Math.round(rect.width), cssH = Math.round(rect.height);
		if (cssW <= 0 || cssH <= 0) return { w: 0, h: 0 };
		const dpr = window.devicePixelRatio || 1;
		const bw = Math.round(cssW * dpr), bh = Math.round(cssH * dpr);
		if (this.canvas.width !== bw || this.canvas.height !== bh) {
			this.canvas.width = bw; this.canvas.height = bh;
			this.canvas.style.width = cssW + 'px';
			this.canvas.style.height = cssH + 'px';
		}
		// Return BACKBUFFER pixel size — render3D feeds this straight into
		// gl.viewport(0, 0, w, h). Passing CSS dimensions on a 2× hidpi
		// display would render into the bottom-left quarter only, which is
		// what "not centered" looked like.
		return { w: bw, h: bh };
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
