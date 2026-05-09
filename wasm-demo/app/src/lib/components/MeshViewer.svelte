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
	let renderer: THREE.WebGLRenderer | null = null;
	let scene: THREE.Scene | null = null;
	let camera: THREE.PerspectiveCamera | null = null;
	let controls: OrbitControls | null = null;
	let resize_observer: ResizeObserver | null = null;
	let surface_meshes: THREE.Mesh[] = [];
	let wire_lines: THREE.LineSegments | null = null;
	let raf_id: number | null = null;

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

	/** Build per-tag THREE.Mesh objects from the surface triangles. */
	function build_geometry(m: MeshData) {
		const by_tag = new Map<number, number[]>();
		const ntri = m.tri_phys.length;
		for (let i = 0; i < ntri; i++) {
			const tag = m.tri_phys[i];
			let arr = by_tag.get(tag);
			if (!arr) { arr = []; by_tag.set(tag, arr); }
			arr.push(m.tris[i * 3], m.tris[i * 3 + 1], m.tris[i * 3 + 2]);
		}

		const meshes: THREE.Mesh[] = [];
		for (const [tag, idx] of by_tag.entries()) {
			const name = m.phys_names.get(tag) ?? `tag_${tag}`;
			if (name === 'abc' || name.startsWith('_mat_')) continue;
			const kind = classify(name);
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
				// Push 2D conductor plates slightly toward the camera so they
				// don't z-fight with the (coplanar) interior face of the oxide box
				polygonOffset: kind === 'conductor' || kind === 'port' || kind === 'gnd',
				polygonOffsetFactor: -2,
				polygonOffsetUnits: -2
			});
			const mesh3 = new THREE.Mesh(geom, mat);
			mesh3.renderOrder = style.render_order;
			mesh3.userData = { tag, name, kind };
			meshes.push(mesh3);
		}
		// Sort so dielectric draws first → conductors → ports
		meshes.sort((a, b) => a.renderOrder - b.renderOrder);
		return meshes;
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

		renderer = new THREE.WebGLRenderer({ antialias: true });
		renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
		container.appendChild(renderer.domElement);

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
		if (!mesh) return [] as { name: string; color: string; tag: number }[];
		const seen = new Set<number>();
		const out: { name: string; color: string; tag: number }[] = [];
		for (const tag of mesh.tri_phys) {
			if (seen.has(tag)) continue;
			const name = mesh.phys_names.get(tag) ?? `tag_${tag}`;
			if (name === 'abc' || name.startsWith('_mat_')) {
				seen.add(tag);
				continue;
			}
			seen.add(tag);
			out.push({ name, color: plotColors.cycle[tag % plotColors.cycle.length], tag });
		}
		return out;
	});
</script>

<div class="viewer">
	<div class="canvas" bind:this={container}></div>

	{#if tag_legend.length > 0}
		<div class="legend">
			{#each tag_legend as l}
				<div class="legend-item">
					<span class="swatch" style="background: {l.color}"></span>
					<span class="legend-name">{l.name}</span>
				</div>
			{/each}
		</div>
	{/if}

	{#if mesh}
		<div class="stats">
			{(mesh.nodes.length / 3) | 0} nodes · {(mesh.tris.length / 3) | 0} tris · {(mesh.tets.length / 4) | 0} tets
		</div>
	{/if}
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
		top: 8px;
		left: 8px;
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
	.legend-item {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.swatch {
		width: 10px;
		height: 10px;
		flex-shrink: 0;
	}
	.legend-name {
		text-transform: none;
	}
	.stats {
		position: absolute;
		bottom: 8px;
		right: 8px;
		background: rgba(24, 24, 29, 0.75);
		border: 1px solid var(--border-subtle);
		padding: 4px 8px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
		pointer-events: none;
	}
</style>
