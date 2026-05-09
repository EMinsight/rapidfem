<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import * as THREE from 'three';
	import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
	import type { MeshData } from '$lib/msh';
	import { palette, plotColors } from '$lib/theme';

	let {
		mesh = null,
		mode = 'geometry'
	}: {
		mesh?: MeshData | null;
		mode?: 'geometry' | 'mesh' | 'both';
	} = $props();

	let container = $state<HTMLDivElement | null>(null);
	let canvas_el: HTMLCanvasElement | null = null;
	let renderer: THREE.WebGLRenderer | null = null;
	let scene: THREE.Scene | null = null;
	let camera: THREE.PerspectiveCamera | null = null;
	let controls: OrbitControls | null = null;
	let resize_observer: ResizeObserver | null = null;
	let surface_meshes: THREE.Mesh[] = [];
	let wire_lines: THREE.LineSegments | null = null;
	let raf_id: number | null = null;
	let z_flip = 1;
	let cursor_world = $state({ x: 0, y: 0, z: 0 });
	let last_bbox: MeshData['bbox'] | null = null;

	// ── Imperative API exposed via bind:this ────────────────────────────────
	export function zoom_in() { dolly(1 / 1.3); }
	export function zoom_out() { dolly(1.3); }
	export function fit_view() { if (mesh) fit_camera(mesh); }
	export function rotate_90() {
		if (!camera || !controls) return;
		const t = controls.target;
		const offset = new THREE.Vector3().subVectors(camera.position, t);
		const ang = Math.PI / 2;
		const cos = Math.cos(ang), sin = Math.sin(ang);
		const x = offset.x * cos - offset.y * sin;
		const y = offset.x * sin + offset.y * cos;
		camera.position.set(t.x + x, t.y + y, camera.position.z);
		camera.lookAt(t);
		controls.update();
	}
	export function flip_z() {
		if (!camera) return;
		z_flip *= -1;
		camera.up.set(0, 0, z_flip);
		controls?.update();
	}
	export function save_png() {
		if (!renderer || !canvas_el || !scene || !camera) return;
		const prev_clear = renderer.getClearColor(new THREE.Color()).clone();
		const prev_alpha = renderer.getClearAlpha();
		renderer.setClearColor(0x000000, 0);
		renderer.render(scene, camera);
		canvas_el.toBlob((blob) => {
			if (!blob) return;
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = 'rapidfem-mesh.png';
			a.click();
			URL.revokeObjectURL(url);
			renderer!.setClearColor(prev_clear, prev_alpha);
		}, 'image/png');
	}

	function dolly(scale: number) {
		if (!camera || !controls) return;
		const offset = new THREE.Vector3().subVectors(camera.position, controls.target);
		offset.multiplyScalar(scale);
		camera.position.copy(controls.target).add(offset);
		controls.update();
	}

	type TagKind = 'dielectric' | 'conductor' | 'port' | 'gnd' | 'other';
	function classify(name: string): TagKind {
		if (name === 'substrate' || name === 'oxide' || name === 'air') return 'dielectric';
		if (name.endsWith('_gnd') || name === 'gnd') return 'gnd';
		if (name === 'p1' || name === 'p2' || /^p\d+$/.test(name)) return 'port';
		// Anything else that's been registered as a face physical group is a
		// conductor (e.g. met5, met4, li1, "wire_pec" from manual examples).
		return 'conductor';
	}

	function style_for(kind: TagKind, tag: number): {
		color: string;
		opacity: number;
		render_order: number;
		double_side: boolean;
		depth_write: boolean;
	} {
		switch (kind) {
			case 'dielectric':
				// Pastel translucent, render first, behind everything else.
				return {
					color:
						{ substrate: '#4a9ec2', oxide: '#7b5e8a', air: '#3a3a44' }[
							{ 0: 'substrate' }[tag] ?? ''
						] ?? '#5a6470',
					opacity: 0.08,
					render_order: 0,
					double_side: false,
					depth_write: false
				};
			case 'gnd':
				return { color: '#4a9ec2', opacity: 0.9, render_order: 1, double_side: true, depth_write: true };
			case 'conductor':
				return { color: '#e8944a', opacity: 1.0, render_order: 2, double_side: true, depth_write: true };
			case 'port':
				return { color: '#d9513c', opacity: 0.85, render_order: 3, double_side: true, depth_write: false };
			default:
				return { color: '#7d7a85', opacity: 0.5, render_order: 1, double_side: true, depth_write: true };
		}
	}

	/** Custom dielectric color by name (substrate=blue, oxide=purple, air=grey). */
	function dielectric_color(name: string): string {
		if (name === 'substrate') return '#4a9ec2';
		if (name === 'oxide') return '#7b5e8a';
		if (name === 'air') return '#5a6470';
		return '#5a6470';
	}

	/** From the tet array, derive the BOUNDARY of each volume (faces that
	 * belong to exactly one tet within that volume — i.e. the volume's
	 * external skin). Returns map: volume_tag → flat array of node indices. */
	function build_volume_boundary(m: MeshData): Map<number, number[]> {
		const out = new Map<number, number[]>();
		const ntets = m.tet_phys.length;
		// face_key (sorted node triple) → { vol_tag, indices [a,b,c] (oriented) }
		// If the same face shows up twice within ONE volume, both occurrences cancel
		// (it's an interior face). If it shows up only once, it's a boundary face.
		const seen = new Map<bigint, { vol: number; tri: [number, number, number] } | null>();
		const enc = (a: number, b: number, c: number): bigint => {
			const s = [a, b, c].sort((x, y) => x - y);
			return (BigInt(s[0]) * 0x100000000n + BigInt(s[1])) * 0x100000000n + BigInt(s[2]);
		};
		for (let t = 0; t < ntets; t++) {
			const v = m.tet_phys[t];
			if (!v) continue;
			const a = m.tets[t * 4], b = m.tets[t * 4 + 1], c = m.tets[t * 4 + 2], d = m.tets[t * 4 + 3];
			// Tet faces: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
			const faces: [number, number, number][] = [
				[a, b, c],
				[a, b, d],
				[a, c, d],
				[b, c, d]
			];
			for (const f of faces) {
				const k = enc(f[0], f[1], f[2]);
				const prev = seen.get(k);
				if (prev === undefined) seen.set(k, { vol: v, tri: f });
				else if (prev !== null && prev.vol === v) seen.set(k, null);
				else seen.set(k, { vol: v, tri: f });
			}
		}
		for (const entry of seen.values()) {
			if (!entry) continue;
			let arr = out.get(entry.vol);
			if (!arr) { arr = []; out.set(entry.vol, arr); }
			arr.push(entry.tri[0], entry.tri[1], entry.tri[2]);
		}
		return out;
	}

	/** Build per-tag THREE.Mesh objects. Two sources:
	 *  - explicit named SURFACE tris (met5, p1, p1_gnd, …) — these are the
	 *    user-tagged 2D plates;
	 *  - implicit VOLUME boundary tris derived from the tet array, colored
	 *    by their volume name (substrate, oxide, …). */
	function build_geometry(m: MeshData) {
		const meshes: THREE.Mesh[] = [];

		// 1) Named SURFACE tris (dim=2 physical groups)
		const by_surf_tag = new Map<number, number[]>();
		for (let i = 0; i < m.tri_phys.length; i++) {
			const tag = m.tri_phys[i];
			if (!tag) continue;             // unnamed surface tri → skip (handled via volume boundary)
			if ((m.phys_dim.get(tag) ?? 2) !== 2) continue;  // only surface groups here
			let arr = by_surf_tag.get(tag);
			if (!arr) { arr = []; by_surf_tag.set(tag, arr); }
			arr.push(m.tris[i * 3], m.tris[i * 3 + 1], m.tris[i * 3 + 2]);
		}
		for (const [tag, idx] of by_surf_tag.entries()) {
			const name = m.phys_names.get(tag) ?? `tag_${tag}`;
			if (name === 'abc' || name.startsWith('_mat_')) continue;
			const kind = classify(name);
			meshes.push(make_mesh(m, idx, tag, name, kind));
		}

		// 2) Implicit volume-boundary tris (substrate / oxide / air hulls)
		const vol_boundaries = build_volume_boundary(m);
		for (const [vol_tag, idx] of vol_boundaries.entries()) {
			const name = m.phys_names.get(vol_tag) ?? `vol_${vol_tag}`;
			if (name.startsWith('_mat_')) continue;  // material-marker volumes (already covered)
			meshes.push(make_mesh(m, idx, vol_tag, name, 'dielectric'));
		}

		meshes.sort((a, b) => a.renderOrder - b.renderOrder);
		return meshes;
	}

	function make_mesh(m: MeshData, idx: number[], tag: number, name: string, kind: TagKind): THREE.Mesh {
		const style = style_for(kind, tag);
		if (kind === 'dielectric') style.color = dielectric_color(name);

		const geom = new THREE.BufferGeometry();
		geom.setAttribute('position', new THREE.BufferAttribute(m.nodes, 3));
		geom.setIndex(new THREE.Uint32BufferAttribute(idx, 1));
		geom.computeVertexNormals();

		const mat = new THREE.MeshStandardMaterial({
			color: new THREE.Color(style.color),
			transparent: style.opacity < 1,
			opacity: style.opacity,
			roughness: kind === 'conductor' ? 0.35 : 0.6,
			metalness: kind === 'conductor' ? 0.6 : 0.05,
			side: style.double_side ? THREE.DoubleSide : THREE.FrontSide,
			depthWrite: style.depth_write,
			polygonOffset: kind === 'conductor' || kind === 'port' || kind === 'gnd',
			polygonOffsetFactor: -2,
			polygonOffsetUnits: -2
		});
		const mesh3 = new THREE.Mesh(geom, mat);
		mesh3.renderOrder = style.render_order;
		mesh3.userData = { tag, name, kind };
		return mesh3;
	}

	function build_wireframe(m: MeshData) {
		// All surface triangle edges (deduplicated)
		const edge_set = new Set<bigint>();
		const edges: number[] = [];
		const add_edge = (a: number, b: number) => {
			const lo = a < b ? a : b;
			const hi = a < b ? b : a;
			const key = (BigInt(lo) << 32n) | BigInt(hi);
			if (!edge_set.has(key)) {
				edge_set.add(key);
				edges.push(a, b);
			}
		};
		for (let i = 0; i < m.tri_phys.length; i++) {
			const a = m.tris[i * 3], b = m.tris[i * 3 + 1], c = m.tris[i * 3 + 2];
			add_edge(a, b); add_edge(b, c); add_edge(c, a);
		}
		const geom = new THREE.BufferGeometry();
		geom.setAttribute('position', new THREE.BufferAttribute(m.nodes, 3));
		geom.setIndex(new THREE.Uint32BufferAttribute(edges, 1));
		const mat = new THREE.LineBasicMaterial({
			color: new THREE.Color(palette.textDim),
			transparent: true,
			opacity: 0.55
		});
		return new THREE.LineSegments(geom, mat);
	}

	function setup() {
		if (!container) return;
		scene = new THREE.Scene();
		scene.background = new THREE.Color(palette.bgInset);

		camera = new THREE.PerspectiveCamera(40, 1, 1e-9, 1);
		camera.up.set(0, 0, 1);

		renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		canvas_el = renderer.domElement;
		container.appendChild(canvas_el);
		canvas_el.addEventListener('pointermove', on_pointer_move);

		controls = new OrbitControls(camera, renderer.domElement);
		controls.enableDamping = true;
		controls.dampingFactor = 0.08;

		const ambient = new THREE.AmbientLight(0xffffff, 0.55);
		scene.add(ambient);
		const dir1 = new THREE.DirectionalLight(0xffffff, 0.7);
		dir1.position.set(1, 1, 1);
		scene.add(dir1);
		const dir2 = new THREE.DirectionalLight(0xffffff, 0.35);
		dir2.position.set(-1, -0.5, 0.7);
		scene.add(dir2);

		resize_observer = new ResizeObserver(handle_resize);
		resize_observer.observe(container);
		handle_resize();

		const tick = () => {
			controls?.update();
			if (renderer && scene && camera) renderer.render(scene, camera);
			raf_id = requestAnimationFrame(tick);
		};
		raf_id = requestAnimationFrame(tick);
	}

	const _ndc_v = new THREE.Vector3();
	const _ray = new THREE.Raycaster();
	const _z0_plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
	const _hit = new THREE.Vector3();
	function on_pointer_move(e: PointerEvent) {
		if (!camera || !canvas_el) return;
		const rect = canvas_el.getBoundingClientRect();
		const x_ndc = ((e.clientX - rect.left) / rect.width) * 2 - 1;
		const y_ndc = -((e.clientY - rect.top) / rect.height) * 2 + 1;
		_ndc_v.set(x_ndc, y_ndc, 0.5);
		_ray.setFromCamera({ x: x_ndc, y: y_ndc } as THREE.Vector2, camera);
		// Project onto z=z_metal plane (use bbox top of metal layers if known, else 0)
		// Use camera target z as the reference plane height
		const z_ref = controls?.target.z ?? 0;
		_z0_plane.constant = -z_ref;
		if (_ray.ray.intersectPlane(_z0_plane, _hit)) {
			cursor_world = { x: _hit.x, y: _hit.y, z: z_ref };
		}
	}

	function handle_resize() {
		if (!container || !renderer || !camera) return;
		const w = container.clientWidth;
		const h = container.clientHeight;
		if (w <= 0 || h <= 0) return;
		renderer.setSize(w, h, false);
		camera.aspect = w / h;
		camera.updateProjectionMatrix();
	}

	function fit_camera(m: MeshData) {
		if (!camera || !controls) return;
		// Compute bbox from interesting faces only (conductors + ports + gnd).
		// Falls back to full mesh bbox if there are no such faces.
		let xmin = Infinity, ymin = Infinity, zmin = Infinity;
		let xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
		let have = false;
		const ntri = m.tri_phys.length;
		for (let i = 0; i < ntri; i++) {
			const name = m.phys_names.get(m.tri_phys[i]) ?? '';
			const kind = classify(name);
			if (kind === 'dielectric' || name === 'abc' || name.startsWith('_mat_')) continue;
			have = true;
			for (let k = 0; k < 3; k++) {
				const ni = m.tris[i * 3 + k] * 3;
				const x = m.nodes[ni], y = m.nodes[ni + 1], z = m.nodes[ni + 2];
				if (x < xmin) xmin = x; if (x > xmax) xmax = x;
				if (y < ymin) ymin = y; if (y > ymax) ymax = y;
				if (z < zmin) zmin = z; if (z > zmax) zmax = z;
			}
		}
		if (!have) {
			[xmin, ymin, zmin] = m.bbox.min;
			[xmax, ymax, zmax] = m.bbox.max;
		}
		const cx = (xmin + xmax) / 2, cy = (ymin + ymax) / 2, cz = (zmin + zmax) / 2;
		const dx = xmax - xmin, dy = ymax - ymin, dz = zmax - zmin;
		const radius = Math.max(0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz), 1e-9);
		controls.target.set(cx, cy, cz);
		camera.position.set(cx + radius * 1.6, cy - radius * 1.6, cz + radius * 1.2);
		// Use the FULL mesh bbox to size the near/far so dielectric volumes stay visible
		const fullDx = m.bbox.max[0] - m.bbox.min[0];
		const fullDy = m.bbox.max[1] - m.bbox.min[1];
		const fullDz = m.bbox.max[2] - m.bbox.min[2];
		const fullRadius = 0.5 * Math.sqrt(fullDx * fullDx + fullDy * fullDy + fullDz * fullDz);
		camera.near = Math.max(fullRadius * 1e-4, 1e-9);
		camera.far = fullRadius * 200;
		camera.updateProjectionMatrix();
		controls.update();
	}

	function clear_objects() {
		if (!scene) return;
		for (const m of surface_meshes) {
			scene.remove(m);
			m.geometry.dispose();
			(m.material as THREE.Material).dispose();
		}
		surface_meshes = [];
		if (wire_lines) {
			scene.remove(wire_lines);
			wire_lines.geometry.dispose();
			(wire_lines.material as THREE.Material).dispose();
			wire_lines = null;
		}
	}

	function rebuild() {
		if (!scene || !mesh) return;
		clear_objects();
		const showFaces = mode === 'geometry' || mode === 'both';
		const showWire = mode === 'mesh' || mode === 'both';
		if (showFaces) {
			surface_meshes = build_geometry(mesh);
			for (const m of surface_meshes) scene.add(m);
		}
		if (showWire) {
			wire_lines = build_wireframe(mesh);
			scene.add(wire_lines);
		}
		fit_camera(mesh);
	}

	$effect(() => {
		mesh; mode;
		if (scene && mesh) rebuild();
	});

	onMount(() => setup());

	onDestroy(() => {
		if (raf_id != null) cancelAnimationFrame(raf_id);
		resize_observer?.disconnect();
		clear_objects();
		renderer?.dispose();
	});

	const tag_legend = $derived.by(() => {
		if (!mesh) return [] as { name: string; color: string; kind: TagKind }[];
		const seen = new Set<number>();
		const items: { name: string; color: string; kind: TagKind; rank: number }[] = [];
		const add = (tag: number, kind: TagKind) => {
			if (seen.has(tag)) return;
			seen.add(tag);
			const name = mesh!.phys_names.get(tag) ?? `tag_${tag}`;
			if (name === 'abc' || name.startsWith('_mat_')) return;
			const style = style_for(kind, tag);
			const color = kind === 'dielectric' ? dielectric_color(name) : style.color;
			const rank =
				kind === 'conductor' ? 0 : kind === 'port' ? 1 : kind === 'gnd' ? 2 : 3;
			items.push({ name, color, kind, rank });
		};
		for (let i = 0; i < mesh.tri_phys.length; i++) {
			const tag = mesh.tri_phys[i];
			if (!tag || (mesh.phys_dim.get(tag) ?? 2) !== 2) continue;
			const name = mesh.phys_names.get(tag) ?? '';
			add(tag, classify(name));
		}
		for (let i = 0; i < mesh.tet_phys.length; i++) {
			const tag = mesh.tet_phys[i];
			const name = mesh.phys_names.get(tag) ?? '';
			if (name.startsWith('_mat_')) continue;
			add(tag, 'dielectric');
		}
		items.sort((a, b) => a.rank - b.rank);
		return items;
	});
</script>

<div class="viewer">
	<div class="canvas" bind:this={container}></div>

	{#if tag_legend.length > 0}
		<div class="legend">
			{#each tag_legend as l}
				<div class="legend-item">
					<span class="swatch" style="background: {l.color}; opacity: {l.kind === 'dielectric' ? 0.4 : 1};"></span>
					<span class="legend-name">{l.name}</span>
				</div>
			{/each}
		</div>
	{/if}

	<div class="viewer-toolbar">
		<button class="tb" onclick={zoom_in}><span class="tip">Zoom in<kbd>+</kbd></span>+</button>
		<button class="tb" onclick={zoom_out}><span class="tip">Zoom out<kbd>−</kbd></span>−</button>
		<button class="tb" onclick={fit_view}>
			<span class="tip">Fit view<kbd>F</kbd></span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
				<polyline points="1,5 1,1 5,1" />
				<polyline points="11,1 15,1 15,5" />
				<polyline points="15,11 15,15 11,15" />
				<polyline points="5,15 1,15 1,11" />
				<rect x="5" y="5" width="6" height="6" rx="0.5" />
			</svg>
		</button>
		<button class="tb" onclick={rotate_90}>
			<span class="tip">Rotate<kbd>R</kbd></span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
				<polyline points="15.3 2.7 15.3 6.7 11.3 6.7" />
				<path d="M13.66 10a6 6 0 1 1-1.41-6.24L15.3 6.7" />
			</svg>
		</button>
		<button class="tb" onclick={flip_z}>
			<span class="tip">Flip Z<kbd>Z</kbd></span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
				<path d="M3 4h10" />
				<path d="M8 4v8" />
				<path d="M5 9l3 3 3-3" />
			</svg>
		</button>
		<button class="tb" onclick={save_png}>
			<span class="tip">Save PNG<kbd>Ctrl+S</kbd></span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
				<path d="M2 10v3h12v-3" />
				<path d="M8 2v8" />
				<path d="M5 7l3 3 3-3" />
			</svg>
		</button>
	</div>

	<div class="hud">
		<span class="coord">x {(cursor_world.x * 1e6).toFixed(1)} µm</span>
		<span class="coord">y {(cursor_world.y * 1e6).toFixed(1)} µm</span>
		{#if mesh}
			<span class="coord stats">{(mesh.nodes.length / 3) | 0}n · {(mesh.tris.length / 3) | 0}t · {(mesh.tets.length / 4) | 0}T</span>
		{/if}
	</div>
</div>

<style>
	.viewer {
		position: relative;
		width: 100%;
		height: 100%;
		min-height: 320px;
		background: var(--bg-inset);
		overflow: hidden;
	}
	.canvas {
		position: absolute;
		inset: 0;
	}
	.legend {
		position: absolute;
		top: 10px;
		left: 10px;
		background: rgba(24, 24, 29, 0.75);
		border: 1px solid var(--border-subtle);
		padding: 6px 8px;
		display: flex;
		flex-direction: column;
		gap: 3px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
		pointer-events: none;
	}
	.legend-item { display: flex; align-items: center; gap: 6px; }
	.swatch { width: 10px; height: 10px; flex-shrink: 0; }

	.viewer-toolbar {
		position: absolute;
		top: 10px;
		right: 10px;
		z-index: 10;
		display: flex;
		gap: 2px;
	}
	.tb {
		position: relative;
		width: 28px;
		height: 28px;
		border: 1px solid var(--border);
		background: var(--bg-surface);
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: 14px;
		font-weight: 600;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 0;
		transition: background var(--transition), border-color var(--transition), color var(--transition);
	}
	.tb:hover {
		background: var(--bg-panel);
		border-color: var(--accent);
		color: var(--text);
	}
	.tb .tip {
		display: none;
		position: absolute;
		top: calc(100% + 6px);
		right: 0;
		white-space: nowrap;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		font-weight: 400;
		color: var(--text-muted);
		background: var(--bg-surface);
		border: 1px solid var(--border);
		padding: 3px 8px;
		pointer-events: none;
		z-index: 20;
	}
	.tb .tip kbd {
		margin-left: 6px;
		color: var(--accent);
		font-family: var(--font-mono);
		font-weight: 600;
	}
	.tb:hover .tip { display: flex; align-items: center; gap: 4px; }

	.hud {
		position: absolute;
		bottom: 8px;
		left: 8px;
		display: flex;
		gap: 12px;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-dim);
		pointer-events: none;
	}
	.hud .stats { color: var(--text-muted); }
</style>
