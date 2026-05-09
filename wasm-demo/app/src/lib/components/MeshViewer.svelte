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

	function color_for_tag(tag: number, _name: string): THREE.Color {
		const c = plotColors.cycle[tag % plotColors.cycle.length];
		return new THREE.Color(c);
	}

	/** Build per-tag THREE.Mesh objects from the surface triangles. */
	function build_geometry(m: MeshData) {
		// Group triangles by their physical tag
		const by_tag = new Map<number, number[]>();
		const ntri = m.tri_phys.length;
		for (let i = 0; i < ntri; i++) {
			const tag = m.tri_phys[i];
			let arr = by_tag.get(tag);
			if (!arr) {
				arr = [];
				by_tag.set(tag, arr);
			}
			arr.push(m.tris[i * 3], m.tris[i * 3 + 1], m.tris[i * 3 + 2]);
		}

		const meshes: THREE.Mesh[] = [];
		for (const [tag, idx] of by_tag.entries()) {
			const name = m.phys_names.get(tag) ?? `tag_${tag}`;
			// Skip ABCs and material-only volume groups in surface viz
			if (name === 'abc' || name.startsWith('_mat_')) continue;
			const geom = new THREE.BufferGeometry();
			geom.setAttribute('position', new THREE.BufferAttribute(m.nodes, 3));
			geom.setIndex(new THREE.Uint32BufferAttribute(idx, 1));
			geom.computeVertexNormals();
			const color = color_for_tag(tag, name);
			const isPort = name === 'p1' || name === 'p2' || name.endsWith('_gnd');
			const mat = new THREE.MeshStandardMaterial({
				color,
				transparent: true,
				opacity: isPort ? 0.6 : 0.85,
				roughness: 0.55,
				metalness: 0.1,
				side: THREE.DoubleSide,
				depthWrite: !isPort
			});
			const mesh3 = new THREE.Mesh(geom, mat);
			mesh3.userData = { tag, name };
			meshes.push(mesh3);
		}
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
		const cx = (m.bbox.min[0] + m.bbox.max[0]) / 2;
		const cy = (m.bbox.min[1] + m.bbox.max[1]) / 2;
		const cz = (m.bbox.min[2] + m.bbox.max[2]) / 2;
		const dx = m.bbox.max[0] - m.bbox.min[0];
		const dy = m.bbox.max[1] - m.bbox.min[1];
		const dz = m.bbox.max[2] - m.bbox.min[2];
		const radius = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
		controls.target.set(cx, cy, cz);
		camera.position.set(cx + radius * 1.6, cy - radius * 1.6, cz + radius * 1.2);
		camera.near = radius * 1e-3;
		camera.far = radius * 100;
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
