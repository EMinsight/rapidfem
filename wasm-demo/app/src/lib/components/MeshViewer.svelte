<script lang="ts">
	import { onMount } from 'svelte';
	import {
		initGL, disposeGL, clearMeshes, addMesh, addLineMesh, setBBox,
		render3D, fitCamera, type GLState, type Camera
	} from '$lib/render/canvas3d';
	import type { MeshData } from '$lib/msh';
	import { palette } from '$lib/theme';

	let {
		mesh = null,
		mode = 'geometry'
	}: {
		mesh?: MeshData | null;
		mode?: 'geometry' | 'mesh' | 'both';
	} = $props();

	let canvas = $state<HTMLCanvasElement | null>(null);
	let container = $state<HTMLDivElement | null>(null);
	let gl_state: GLState | null = null;
	let camera: Camera = { theta: Math.PI / 4, phi: Math.PI / 4, distance: 1, target: [0, 0, 0] };
	let z_flip = 1;
	let mounted = false;
	let needs_rebuild = true;
	let cursor_world = $state({ x: 0, y: 0 });
	let is_dragging = false;
	let is_right_drag = false;
	let last_mouse = { x: 0, y: 0 };

	// ── Imperative API ─────────────────────────────────────────────────
	export function zoom_in() { camera = { ...camera, distance: camera.distance / 1.3 }; render_frame(); }
	export function zoom_out() { camera = { ...camera, distance: camera.distance * 1.3 }; render_frame(); }
	export function fit_view() {
		if (!mesh) return;
		camera = fitCamera(mesh.bbox.min, mesh.bbox.max);
		render_frame();
	}
	export function rotate_90() {
		camera = { ...camera, theta: camera.theta + Math.PI / 2 };
		render_frame();
	}
	export function flip_z() {
		z_flip *= -1;
		render_frame();
	}
	export function save_png() {
		if (!canvas) return;
		render_frame();
		canvas.toBlob((blob) => {
			if (!blob) return;
			const url = URL.createObjectURL(blob);
			const a = document.createElement('a');
			a.href = url;
			a.download = 'rapidfem-mesh.png';
			a.click();
			URL.revokeObjectURL(url);
		}, 'image/png');
	}

	// ── Mesh classification & coloring ──────────────────────────────────
	type Kind = 'dielectric' | 'conductor' | 'port' | 'gnd';
	function classify(name: string): Kind | null {
		if (name === 'abc' || name.startsWith('_mat_')) return null;
		if (name === 'substrate' || name === 'oxide' || name === 'air') return 'dielectric';
		if (name.endsWith('_gnd') || name === 'gnd') return 'gnd';
		if (name === 'p1' || name === 'p2' || /^p\d+$/.test(name)) return 'port';
		return 'conductor';
	}

	function color_for(kind: Kind, name: string): [number, number, number] {
		// Match rapidpassives palette where it makes sense (accent for ports,
		// muted dielectrics, copper-orange for conductors).
		if (kind === 'dielectric') {
			if (name === 'substrate') return hex(palette.accentSecondary === '#e8944a' ? '#4a9ec2' : '#4a9ec2');
			if (name === 'oxide') return hex('#7b5e8a');
			if (name === 'air') return hex('#5a6470');
			return hex('#5a6470');
		}
		if (kind === 'gnd') return hex('#5aad78');     // greenish ground
		if (kind === 'port') return hex(palette.accent);
		return hex(palette.accentSecondary);             // conductors → copper-orange
	}

	function hex(s: string): [number, number, number] {
		return [
			parseInt(s.slice(1, 3), 16) / 255,
			parseInt(s.slice(3, 5), 16) / 255,
			parseInt(s.slice(5, 7), 16) / 255
		];
	}

	// ── Build per-group triangle buffers ──────────────────────────────
	function compute_normals(positions: Float32Array): Float32Array {
		const n = positions.length / 9;
		const normals = new Float32Array(n * 9);
		const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0];
		for (let t = 0; t < n; t++) {
			const i = t * 9;
			a[0] = positions[i + 0]; a[1] = positions[i + 1]; a[2] = positions[i + 2];
			b[0] = positions[i + 3]; b[1] = positions[i + 4]; b[2] = positions[i + 5];
			c[0] = positions[i + 6]; c[1] = positions[i + 7]; c[2] = positions[i + 8];
			const e1x = b[0] - a[0], e1y = b[1] - a[1], e1z = b[2] - a[2];
			const e2x = c[0] - a[0], e2y = c[1] - a[1], e2z = c[2] - a[2];
			let nx = e1y * e2z - e1z * e2y;
			let ny = e1z * e2x - e1x * e2z;
			let nz = e1x * e2y - e1y * e2x;
			const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
			nx /= l; ny /= l; nz /= l;
			for (let k = 0; k < 3; k++) {
				normals[i + k * 3 + 0] = nx;
				normals[i + k * 3 + 1] = ny;
				normals[i + k * 3 + 2] = nz;
			}
		}
		return normals;
	}

	/** Volume hull from tets — face appearing exactly once per volume = boundary. */
	function build_volume_boundaries(m: MeshData): Map<number, number[]> {
		const out = new Map<number, number[]>();
		const seen = new Map<bigint, { vol: number; tri: [number, number, number] } | null>();
		const enc = (a: number, b: number, c: number): bigint => {
			const s = [a, b, c].sort((x, y) => x - y);
			return (BigInt(s[0]) * 0x100000000n + BigInt(s[1])) * 0x100000000n + BigInt(s[2]);
		};
		const ntets = m.tet_phys.length;
		for (let t = 0; t < ntets; t++) {
			const v = m.tet_phys[t];
			if (!v) continue;
			const a = m.tets[t * 4], b = m.tets[t * 4 + 1], c = m.tets[t * 4 + 2], d = m.tets[t * 4 + 3];
			const faces: [number, number, number][] = [
				[a, b, c], [a, b, d], [a, c, d], [b, c, d]
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

	function rebuild() {
		if (!gl_state || !mesh) return;
		clearMeshes(gl_state);

		const showFaces = mode === 'geometry' || mode === 'both';
		const showWire = mode === 'mesh' || mode === 'both';

		setBBox(gl_state, mesh.bbox.min, mesh.bbox.max);

		if (showFaces) {
			// 1) Named surface tris
			const by_surf = new Map<number, number[]>();
			for (let i = 0; i < mesh.tri_phys.length; i++) {
				const tag = mesh.tri_phys[i];
				if (!tag || (mesh.phys_dim.get(tag) ?? 2) !== 2) continue;
				let arr = by_surf.get(tag);
				if (!arr) { arr = []; by_surf.set(tag, arr); }
				arr.push(mesh.tris[i * 3], mesh.tris[i * 3 + 1], mesh.tris[i * 3 + 2]);
			}
			for (const [tag, idx] of by_surf.entries()) {
				const name = mesh.phys_names.get(tag) ?? '';
				const kind = classify(name);
				if (!kind) continue;
				push_group(idx, kind, name);
			}
			// 2) Implicit volume hulls
			const vol_b = build_volume_boundaries(mesh);
			for (const [vtag, idx] of vol_b.entries()) {
				const name = mesh.phys_names.get(vtag) ?? '';
				if (!name || name.startsWith('_mat_')) continue;
				push_group(idx, 'dielectric', name);
			}
		}

		if (showWire) {
			// Wireframe edges from surface tris (deduped)
			const edges: number[] = [];
			const seen = new Set<bigint>();
			const add_edge = (a: number, b: number) => {
				const lo = a < b ? a : b;
				const hi = a < b ? b : a;
				const k = (BigInt(lo) << 32n) | BigInt(hi);
				if (!seen.has(k)) {
					seen.add(k);
					edges.push(
						mesh.nodes[a * 3], mesh.nodes[a * 3 + 1], mesh.nodes[a * 3 + 2],
						mesh.nodes[b * 3], mesh.nodes[b * 3 + 1], mesh.nodes[b * 3 + 2]
					);
				}
			};
			for (let i = 0; i < mesh.tri_phys.length; i++) {
				const a = mesh.tris[i * 3], b = mesh.tris[i * 3 + 1], c = mesh.tris[i * 3 + 2];
				add_edge(a, b); add_edge(b, c); add_edge(c, a);
			}
			addLineMesh(gl_state, new Float32Array(edges), hex(palette.textDim));
		}

		needs_rebuild = false;
	}

	function push_group(idx: number[], kind: Kind, name: string) {
		if (!gl_state || !mesh) return;
		const ntri = idx.length / 3;
		if (ntri === 0) return;
		const positions = new Float32Array(ntri * 9);
		for (let t = 0; t < ntri; t++) {
			for (let v = 0; v < 3; v++) {
				const ni = idx[t * 3 + v] * 3;
				positions[t * 9 + v * 3 + 0] = mesh.nodes[ni];
				positions[t * 9 + v * 3 + 1] = mesh.nodes[ni + 1];
				positions[t * 9 + v * 3 + 2] = mesh.nodes[ni + 2];
			}
		}
		const normals = compute_normals(positions);
		addMesh(gl_state, positions, normals, color_for(kind, name));
	}

	// ── Frame loop / sizing ─────────────────────────────────────────────
	function get_size(): { w: number; h: number } {
		if (!container) return { w: 0, h: 0 };
		const r = container.getBoundingClientRect();
		return { w: Math.round(r.width), h: Math.round(r.height) };
	}
	function sync_canvas(): { w: number; h: number } {
		const { w, h } = get_size();
		if (w <= 0 || h <= 0 || !canvas) return { w, h };
		const dpr = window.devicePixelRatio || 1;
		const bw = Math.round(w * dpr), bh = Math.round(h * dpr);
		if (canvas.width !== bw || canvas.height !== bh) {
			canvas.width = bw;
			canvas.height = bh;
			canvas.style.width = w + 'px';
			canvas.style.height = h + 'px';
		}
		return { w: bw, h: bh };
	}
	function render_frame() {
		if (!gl_state || !canvas) return;
		const { w, h } = sync_canvas();
		if (w <= 0 || h <= 0) return;
		if (needs_rebuild) rebuild();
		render3D(gl_state, camera, w, h, z_flip);
	}

	// ── Pointer / wheel handlers (orbit/pan/zoom analog rapidpassives) ──
	function on_wheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
		camera = { ...camera, distance: camera.distance * factor };
		render_frame();
	}
	function on_pointer_down(e: PointerEvent) {
		is_dragging = true;
		is_right_drag = e.button === 2;
		last_mouse = { x: e.clientX, y: e.clientY };
		canvas?.setPointerCapture(e.pointerId);
	}
	function on_pointer_move(e: PointerEvent) {
		// HUD coords (project to z=target plane)
		if (canvas) {
			const r = canvas.getBoundingClientRect();
			const mx = e.clientX - r.left, my = e.clientY - r.top;
			const { w, h } = get_size();
			const halfH = camera.distance * Math.tan(Math.PI / 12);
			const halfW = halfH * (w / h || 1);
			const nx = (mx / w - 0.5) * 2;
			const ny = -(my / h - 0.5) * 2;
			const ct = Math.cos(camera.theta), st = Math.sin(camera.theta);
			cursor_world = {
				x: camera.target[0] + nx * halfW * ct + ny * halfH * st * Math.sin(camera.phi),
				y: camera.target[1] - nx * halfW * st + ny * halfH * ct * Math.sin(camera.phi)
			};
		}
		if (!is_dragging) return;
		const dx = e.clientX - last_mouse.x;
		const dy = e.clientY - last_mouse.y;
		last_mouse = { x: e.clientX, y: e.clientY };
		if (is_right_drag) {
			const panScale = camera.distance * 0.0007;
			const ct = Math.cos(camera.theta), st = Math.sin(camera.theta);
			camera = {
				...camera,
				target: [
					camera.target[0] + (dx * ct - dy * st * Math.sin(camera.phi)) * panScale,
					camera.target[1] - (dx * st + dy * ct * Math.sin(camera.phi)) * panScale,
					camera.target[2] + dy * Math.cos(camera.phi) * panScale
				]
			};
		} else {
			camera = {
				...camera,
				theta: camera.theta + dx * 0.005,
				phi: Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, camera.phi + dy * 0.005))
			};
		}
		render_frame();
	}
	function on_pointer_up() { is_dragging = false; is_right_drag = false; }
	function on_context_menu(e: Event) { e.preventDefault(); }
	function on_dbl_click() { fit_view(); }

	// ── Lifecycle ───────────────────────────────────────────────────────
	onMount(() => {
		mounted = true;
		if (!canvas) return;
		gl_state = initGL(canvas);
		if (!gl_state) return;

		const ro = new ResizeObserver(() => mounted && render_frame());
		if (container) ro.observe(container);

		// Initial fit + render once mesh is available
		if (mesh) {
			camera = fitCamera(mesh.bbox.min, mesh.bbox.max);
			needs_rebuild = true;
		}
		requestAnimationFrame(render_frame);

		return () => {
			mounted = false;
			ro.disconnect();
			if (gl_state) disposeGL(gl_state);
			gl_state = null;
		};
	});

	// React to mesh / mode changes
	$effect(() => {
		mesh; mode;
		if (!mounted || !gl_state) return;
		if (mesh) camera = fitCamera(mesh.bbox.min, mesh.bbox.max);
		needs_rebuild = true;
		render_frame();
	});

	const tag_legend = $derived.by(() => {
		if (!mesh) return [] as { name: string; color: string; kind: Kind; rank: number }[];
		const seen = new Set<number>();
		const items: { name: string; color: string; kind: Kind; rank: number }[] = [];
		const add = (tag: number, kind: Kind) => {
			if (seen.has(tag)) return;
			seen.add(tag);
			const name = mesh!.phys_names.get(tag) ?? '';
			if (!name || name === 'abc' || name.startsWith('_mat_')) return;
			const c = color_for(kind, name);
			const rank = kind === 'conductor' ? 0 : kind === 'port' ? 1 : kind === 'gnd' ? 2 : 3;
			items.push({
				name,
				color: `rgb(${(c[0] * 255) | 0},${(c[1] * 255) | 0},${(c[2] * 255) | 0})`,
				kind,
				rank
			});
		};
		for (let i = 0; i < mesh.tri_phys.length; i++) {
			const tag = mesh.tri_phys[i];
			if (!tag || (mesh.phys_dim.get(tag) ?? 2) !== 2) continue;
			const k = classify(mesh.phys_names.get(tag) ?? '');
			if (k) add(tag, k);
		}
		for (let i = 0; i < mesh.tet_phys.length; i++) {
			const tag = mesh.tet_phys[i];
			const name = mesh.phys_names.get(tag) ?? '';
			if (name && !name.startsWith('_mat_')) add(tag, 'dielectric');
		}
		items.sort((a, b) => a.rank - b.rank);
		return items;
	});
</script>

<div class="viewer" bind:this={container}>
	<canvas
		bind:this={canvas}
		onwheel={on_wheel}
		onpointerdown={on_pointer_down}
		onpointermove={on_pointer_move}
		onpointerup={on_pointer_up}
		oncontextmenu={on_context_menu}
		ondblclick={on_dbl_click}
	></canvas>

	{#if tag_legend.length > 0}
		<div class="legend">
			{#each tag_legend as l}
				<div class="legend-item">
					<span class="swatch" style="background: {l.color}; opacity: {l.kind === 'dielectric' ? 0.5 : 1};"></span>
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
				<polyline points="1,5 1,1 5,1" /><polyline points="11,1 15,1 15,5" />
				<polyline points="15,11 15,15 11,15" /><polyline points="5,15 1,15 1,11" />
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
				<path d="M3 4h10" /><path d="M8 4v8" /><path d="M5 9l3 3 3-3" />
			</svg>
		</button>
		<button class="tb" onclick={save_png}>
			<span class="tip">Save PNG<kbd>Ctrl+S</kbd></span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
				<path d="M2 10v3h12v-3" /><path d="M8 2v8" /><path d="M5 7l3 3 3-3" />
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
		background: var(--canvas-bg);
		overflow: hidden;
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
		cursor: grab;
	}
	canvas:active { cursor: grabbing; }

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
	.tb:hover { background: var(--bg-panel); border-color: var(--accent); color: var(--text); }
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
	.tb .tip kbd { margin-left: 6px; color: var(--accent); font-weight: 600; }
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
