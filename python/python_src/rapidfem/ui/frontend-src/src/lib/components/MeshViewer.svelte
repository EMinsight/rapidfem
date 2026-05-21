<script lang="ts">
	import { onMount, untrack } from 'svelte';
	import {
		initGL, disposeGL, clearMeshes, addMesh, addLineMesh, setBBox,
		setPointCloud, setPointPhase, setPointRange, setPointScaleMode,
		render3D, fitCamera, setTagVisible,
		type GLState, type Camera
	} from '$lib/render/canvas3d';
	import type { MeshData } from '$lib/msh';
	import type { TdTrajectoryPayload } from '$lib/api';
	import { palette } from '$lib/theme';
	import { viz_load_mesh, viz_sample } from '$lib/api';

	const EMPTY_F32 = new Float32Array(0);
	// Decade span of the time-domain field cloud's logarithmic colour scale.
	const TD_LOG_DECADES = 3;

	let {
		mesh = null as MeshData | null,
		wireframe = null as { entities: Array<{ name: string; color: [number, number, number]; lines: number[]; tag: number }>; bbox: { min: [number, number, number]; max: [number, number, number] } } | null,
		show_geometry = true,
		show_wireframe = false,
		show_field = false,
		field = null,
		field_channel = $bindable('E' as 'E' | 'J' | 'H'),
		available_channels = ['E'] as ('E' | 'J' | 'H')[],
		point_density = 5,
		scale_mode = $bindable('lin' as 'log' | 'lin'),
		animate_field = false,
		anim_speed = 1,
		// Time-domain field animation: a TdTrajectory point cloud, the frame
		// to render (the notebook page owns the slider / play loop), and the
		// E/H channel switch.
		td_trajectory = null as TdTrajectoryPayload | null,
		td_frame = 0,
		td_channel = $bindable('E' as 'E' | 'H')
	}: {
		mesh?: MeshData | null;
		wireframe?: { entities: Array<{ name: string; color: [number, number, number]; lines: number[]; tag: number }>; bbox: { min: [number, number, number]; max: [number, number, number] } } | null;
		show_geometry?: boolean;
		show_wireframe?: boolean;
		show_field?: boolean;
		field?: Float32Array | null;
		field_channel?: 'E' | 'J' | 'H';
		available_channels?: ('E' | 'J' | 'H')[];
		point_density?: number;
		scale_mode?: 'log' | 'lin';
		animate_field?: boolean;
		anim_speed?: number;
		td_trajectory?: TdTrajectoryPayload | null;
		td_frame?: number;
		td_channel?: 'E' | 'H';
	} = $props();

	// Channel metadata for the colourbar — title + SI unit per channel.
	const CHANNEL_META: Record<'E' | 'J' | 'H', { sym: string; unit: string }> = {
		E: { sym: '|E|', unit: 'V/m'  },
		J: { sym: '|J|', unit: 'A/m²' },
		H: { sym: '|H|', unit: 'A/m'  },
	};

	let canvas = $state<HTMLCanvasElement | null>(null);
	let container = $state<HTMLDivElement | null>(null);
	let gl_state: GLState | null = null;
	let camera: Camera = { theta: Math.PI / 4, phi: Math.PI / 4, distance: 1, target: [0, 0, 0] };
	let z_flip = 1;
	let mounted = false;
	let needs_rebuild = true;
	let cursor_world = $state({ x: 0, y: 0 });
	// Stored as "hidden" so that newly-built meshes (e.g. wireframe after the
	// user toggles Mesh on) default to visible without losing the explicit
	// hides the user picked from the legend.
	let hidden_tags = $state(new Set<number>());
	let field_range = $state<{ min: number; max: number; decades: number } | null>(null);


	function toggle_tag(tag: number) {
		if (!gl_state) return;
		const next = new Set(hidden_tags);
		if (next.has(tag)) next.delete(tag); else next.add(tag);
		hidden_tags = next;
		setTagVisible(gl_state, tag, !next.has(tag));
		schedule_render();
	}
	let is_dragging = false;
	let is_right_drag = false;
	let last_mouse = { x: 0, y: 0 };

	// ── Camera animation (ease-out cubic) ──────────────────────────────
	let anim_id = 0;
	let anim_target: Camera | null = null;
	function effective_camera(): Camera { return anim_target ?? camera; }
	function animate_camera(target: Camera, durationMs = 300) {
		anim_target = target;
		const start = { ...camera, target: [...camera.target] as [number, number, number] };
		const t0 = performance.now();
		const id = ++anim_id;
		function tick() {
			if (!mounted || id !== anim_id) return;
			const t = Math.min(1, (performance.now() - t0) / durationMs);
			const e = 1 - Math.pow(1 - t, 3);
			camera = {
				theta: start.theta + (target.theta - start.theta) * e,
				phi: start.phi + (target.phi - start.phi) * e,
				distance: start.distance + (target.distance - start.distance) * e,
				target: [
					start.target[0] + (target.target[0] - start.target[0]) * e,
					start.target[1] + (target.target[1] - start.target[1]) * e,
					start.target[2] + (target.target[2] - start.target[2]) * e
				]
			};
			schedule_render();
			if (t < 1) requestAnimationFrame(tick);
			else anim_target = null;
		}
		requestAnimationFrame(tick);
	}

	// ── Imperative API ─────────────────────────────────────────────────
	export function zoom_in() {
		const base = effective_camera();
		animate_camera({ ...base, target: [...base.target] as [number, number, number], distance: base.distance / 1.3 }, 200);
	}
	export function zoom_out() {
		const base = effective_camera();
		animate_camera({ ...base, target: [...base.target] as [number, number, number], distance: base.distance * 1.3 }, 200);
	}
	export function fit_view() {
		if (td_trajectory) {
			animate_camera(fitCamera(td_trajectory.bbox.min, td_trajectory.bbox.max), 350);
			return;
		}
		if (!mesh) return;
		animate_camera(fitCamera(mesh.bbox.min, mesh.bbox.max), 350);
	}
	export function rotate_90() {
		const base = effective_camera();
		animate_camera({ ...base, target: [...base.target] as [number, number, number], theta: base.theta + Math.PI / 2 }, 400);
	}
	export function flip_z() {
		z_flip *= -1;
		schedule_render();
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

	/** Strip a trailing `_<digits>` index suffix so "air_1", "dielectric_2"
	 *  classify the same as "air", "dielectric". Returns the base name. */
	function base(name: string): string {
		return name.replace(/_\d+$/, '');
	}

	/** Render a physical-group name into a human-readable legend label.
	 *  Drops the per-class index for singletons (Air, PEC) and keeps it
	 *  for repeatable kinds (Port 1, Port 2, Dielectric 1, Dielectric 2). */
	function pretty_label(name: string): string {
		const b = base(name);
		const m = name.match(/_(\d+)$/);
		const idx = m ? parseInt(m[1], 10) : 0;
		const display: Record<string, string> = {
			air: 'Air',
			conductor: 'Conductor',
			dielectric: 'Dielectric',
			anisotropic: 'Anisotropic',
			port: 'Port',
			pec: 'PEC',
			pmc: 'PMC',
			abc: 'ABC',
			pml: 'PML',
			surfaceimpedance: 'Surface Impedance',
			lumpedelement: 'Lumped Element',
		};
		const d = display[b];
		if (!d) return name;
		// Repeatable kinds keep their index; conceptually-unique ones drop it.
		const repeatable = b === 'port' || b === 'dielectric' || b === 'anisotropic' || b === 'pml';
		return repeatable && idx > 0 ? `${d} ${idx}` : d;
	}

	function classify(name: string): Kind | null {
		const b = base(name);
		// Object-API material names — emitted by geometry.py as
		// "<class_lower>_<idx>" (air, dielectric, conductor, anisotropic).
		if (b === 'air' || b === 'dielectric' || b === 'anisotropic') return 'dielectric';
		if (b === 'conductor') return 'conductor';   // bulk metal volume
		// Object-API physics names — driven ports share a "port_<N>" prefix.
		if (b === 'port') return 'port';
		if (b === 'pec' || b === 'pmc' || b === 'surfaceimpedance' || b === 'lumpedelement') return 'conductor';
		if (b === 'abc') return null;                // absorbing → transparent
		if (b === 'pml') return 'dielectric';
		// Legacy (rfic.Stack / builder string-named physical groups).
		if (name.startsWith('_mat_')) return null;
		if (name === 'substrate' || name === 'oxide') return 'dielectric';
		if (name.endsWith('_gnd') || name === 'gnd' || name === 'ground') return 'gnd';
		if (name === 'p1' || name === 'p2' || /^p\d+$/.test(name) || name.endsWith('_port')) return 'port';
		return 'conductor';
	}

	// Dielectric cycle — distinct hues so multiple dielectrics are
	// distinguishable. Keep air on its own neutral gray channel.
	const DIELECTRIC_CYCLE = ['#4a9ec2', '#6bbf8a', '#7b5e8a', '#a78bd9', '#c4c46b'];

	function color_for(kind: Kind, name: string): [number, number, number] {
		const b = base(name);
		// Materials get type-specific colors regardless of kind classification.
		if (b === 'air') return hex('#5a5a62');                   // neutral gray
		if (b === 'conductor') return hex(palette.accentSecondary); // bulk metal → signature yellow
		if (b === 'dielectric' || b === 'anisotropic') {
			const m = name.match(/_(\d+)$/);
			const idx = m ? Math.max(0, parseInt(m[1], 10) - 1) : 0;
			return hex(DIELECTRIC_CYCLE[idx % DIELECTRIC_CYCLE.length]);
		}
		if (b === 'pml') return hex('#7b5e8a');                   // muted purple
		// Physics objects.
		if (kind === 'port') return hex(palette.accent);          // lava
		if (kind === 'conductor') return hex(palette.accentSecondary);
		if (kind === 'gnd') return hex('#5aad78');
		if (kind === 'dielectric') return hex('#5a5a62');
		// Legacy rfic-style explicit layer names.
		const fixed: Record<string, string> = {
			met5: '#e8944a', met4: '#f0b86a', met3: '#c4c46b',
			met2: '#9bc28b', met1: '#7b9fb8', li1: '#5a8caa',
			via5: '#d9513c', via4: '#e5634f', via3: '#bf4233',
			via2: '#9d3526', via1: '#7c281b', mcon: '#aa6b40',
		};
		return hex(fixed[name] ?? palette.accentSecondary);
	}

	function hex(s: string): [number, number, number] {
		return [
			parseInt(s.slice(1, 3), 16) / 255,
			parseInt(s.slice(3, 5), 16) / 255,
			parseInt(s.slice(5, 7), 16) / 255
		];
	}

	/** The 12 edges of an axis-aligned box as a flat line-segment buffer
	 *  (2 verts × 3 floats per edge) — the spatial-reference frame for the
	 *  time-domain field cloud. */
	function bbox_edges(mn: number[], mx: number[]): number[] {
		const c = [
			[mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
			[mx[0], mx[1], mn[2]], [mn[0], mx[1], mn[2]],
			[mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
			[mx[0], mx[1], mx[2]], [mn[0], mx[1], mx[2]],
		];
		const edges = [
			[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
			[6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
		];
		const out: number[] = [];
		for (const [a, b] of edges) out.push(...c[a], ...c[b]);
		return out;
	}

	/** Phasor-buffer encoding of a static scalar field for the point shader.
	 *  The shader composites |E(t)|² = A·cos²φ + B·sin²φ − 2C·cosφ·sinφ;
	 *  feeding (s², s², 0) makes that collapse to s² for every phase, so a
	 *  time-domain snapshot reuses the frequency-domain cloud unchanged.
	 *  Frame values arrive quantised to 0…1000 of `field_max` — rescale. */
	function td_abc_for_frame(traj: TdTrajectoryPayload, frame: number, channel: 'E' | 'H'): Float32Array {
		const frames = channel === 'H' ? traj.frames_h : traj.frames_e;
		const f = frames[Math.max(0, Math.min(frames.length - 1, frame))] ?? [];
		const scale = (channel === 'H' ? traj.field_max.H : traj.field_max.E) / 1000;
		const abc = new Float32Array(f.length * 3);
		for (let i = 0; i < f.length; i++) {
			const s = f[i] * scale;
			const s2 = s * s;
			abc[i * 3] = s2;
			abc[i * 3 + 1] = s2;
		}
		return abc;
	}

	// (compute_normals is inlined in push_group below to keep the cross-product
	//  in full f64 precision against the original mesh.nodes Float64Array)

	/** Volume hull from tets — for EACH volume independently: face appearing
	 *  exactly once in that volume's tets = part of its hull. A face shared
	 *  between two volumes ends up in BOTH hulls so hiding one volume still
	 *  shows the interface from the other side.
	 *
	 *  CRITICAL: every boundary triangle is oriented so its face normal points
	 *  AWAY from the tet's fourth vertex (= outward from the volume). Without
	 *  this, adjacent boundary triangles can have flipped normals → dappled
	 *  shading on flat surfaces. */
	function build_volume_boundaries(m: MeshData): Map<number, number[]> {
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

		// Orient triangle (a,b,c) so its normal points away from the opposite
		// vertex `o` of the same tet. Returns the (possibly swapped) tri.
		const orient_outward = (
			a: number, b: number, c: number, o: number
		): [number, number, number] => {
			if (!mesh) return [a, b, c];
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
			// If normal · (o - a) > 0, normal points toward o (inward) → swap b/c
			if (nx * dx + ny * dy + nz * dz > 0) return [a, c, b];
			return [a, b, c];
		};

		const out = new Map<number, number[]>();
		for (const [vol, tet_indices] of per_vol.entries()) {
			const seen = new Map<bigint, { count: number; tri: [number, number, number] }>();
			for (const t of tet_indices) {
				const a = m.tets[t * 4], b = m.tets[t * 4 + 1], c = m.tets[t * 4 + 2], d = m.tets[t * 4 + 3];
				// face, opposite vertex
				const tri_descs: [[number, number, number], number][] = [
					[[a, b, c], d],
					[[a, b, d], c],
					[[a, c, d], b],
					[[b, c, d], a]
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

	function rebuild() {
		if (!gl_state) return;
		clearMeshes(gl_state);
		// Time-domain field animation: a point cloud with a faint bounding-box
		// wireframe for spatial reference. The cloud itself is uploaded
		// per-frame by the dedicated $effect below; clearMeshes leaves the
		// point cloud intact.
		// Time-domain trajectory with no geometry of its own (ProblemTD.box):
		// the bounding box is the only spatial frame; the cloud is uploaded
		// per-frame by the dedicated $effect below.
		if (td_trajectory && !mesh) {
			const bb = td_trajectory.bbox;
			setBBox(gl_state, bb.min, bb.max);
			field_norm = null;
			in_field_mode = false;
			addLineMesh(
				gl_state,
				Float32Array.from(bbox_edges(bb.min, bb.max)),
				hex('#3a3a44'), -1,
			);
			needs_rebuild = false;
			return;
		}
		// A trajectory WITH geometry falls through to the mesh path below —
		// the Geometry / Mesh toggles compose freely under the cloud.
		// Wireframe-only view (geometry shown before any g.mesh() call).
		if (!mesh && wireframe && wireframe.entities.length > 0) {
			setBBox(gl_state, wireframe.bbox.min, wireframe.bbox.max);
			field_norm = null;
			in_field_mode = false;
			for (const e of wireframe.entities) {
				if (!e.lines || e.lines.length === 0) continue;
				const c = e.color as [number, number, number];
				addLineMesh(gl_state, Float32Array.from(e.lines), c, e.tag);
			}
			// preserve hidden_tags semantics
			const all_tags = new Set<number>();
			for (const m of gl_state.lineMeshes) all_tags.add(m.tag);
			const cur = untrack(() => hidden_tags);
			for (const m of gl_state.lineMeshes) setTagVisible(gl_state, m.tag, !cur.has(m.tag));
			needs_rebuild = false;
			return;
		}
		if (!mesh) return;

		// Three independent toggles — geometry, wireframe, field — composed
		// freely. The field cloud only renders when both `show_field` and
		// actual field data are present.
		// A TD trajectory owns the point cloud; the FD field path stays off.
		const useField = show_field && field != null && !td_trajectory;
		const showFaces = show_geometry;
		const showWire = show_wireframe;

		setBBox(gl_state, mesh.bbox.min, mesh.bbox.max);
		field_norm = null;
		in_field_mode = useField;

		if (showFaces) {
			// 1) Named surface tris (conductors/ports/gnd). In field mode we
			//    keep these as faint silhouettes for spatial reference.
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
				push_group(idx, kind, name, tag);
			}
			// 2) Implicit volume hulls (substrate/oxide/air, PML, …)
			const vol_b = build_volume_boundaries(mesh);
			for (const [vtag, idx] of vol_b.entries()) {
				const name = mesh.phys_names.get(vtag) ?? '';
				if (!name) continue;
				const kind = classify(name);
				if (!kind) continue;     // e.g. ABC → transparent
				push_group(idx, kind, name, vtag);
			}
		}

		if (showWire) {
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
			// Mesh wireframe: a dimmer grey than the new text-dim palette
			// value so it reads as structure on the dark canvas, not as text.
			addLineMesh(gl_state, Float32Array.from(edges), hex('#3a3a44'), -1);
		}

		// Field splat cloud is sampled asynchronously (see the dedicated
		// `$effect` below). rebuild() just clears any stale cloud here; the
		// sampler repopulates it.
		// A TD trajectory keeps its own point cloud (owned by the per-frame
		// $effect) and gets a faint bounding-box frame; only a stale FD
		// cloud is cleared here.
		if (td_trajectory) {
			addLineMesh(
				gl_state,
				Float32Array.from(bbox_edges(td_trajectory.bbox.min, td_trajectory.bbox.max)),
				hex('#3a3a44'), -1,
			);
		} else if (!useField) {
			setPointCloud(gl_state, EMPTY_F32, EMPTY_F32);
			field_range = null;
		}

		// Re-apply the user's explicit hides to the freshly-built meshes.
		// untrack() so reading hidden_tags here doesn't make the parent
		// rebuild $effect depend on it (which would loop when we write below).
		const all_tags = new Set<number>();
		for (const m of gl_state.meshes) all_tags.add(m.tag);
		for (const m of gl_state.lineMeshes) all_tags.add(m.tag);
		const cur = untrack(() => hidden_tags);
		const next = new Set<number>();
		for (const t of cur) if (all_tags.has(t)) next.add(t);
		// Only assign when something actually dropped out so we don't
		// thrash state with structurally-equal new Sets.
		if (next.size !== cur.size) hidden_tags = next;
		const eff = next.size !== cur.size ? next : cur;
		for (const m of gl_state.meshes) setTagVisible(gl_state, m.tag, !eff.has(m.tag));
		for (const m of gl_state.lineMeshes) setTagVisible(gl_state, m.tag, !eff.has(m.tag));

		needs_rebuild = false;
	}

	let field_norm: Float32Array | null = null;
	let in_field_mode = false;

	function push_group(idx: number[], kind: Kind, name: string, tag: number) {
		if (!gl_state || !mesh) return;
		if (idx.length === 0) return;

		const ntri = idx.length / 3;
		const pos64 = new Float64Array(ntri * 9);
		for (let t = 0; t < ntri; t++) {
			for (let v = 0; v < 3; v++) {
				const ni = idx[t * 3 + v] * 3;
				pos64[t * 9 + v * 3 + 0] = mesh.nodes[ni];
				pos64[t * 9 + v * 3 + 1] = mesh.nodes[ni + 1];
				pos64[t * 9 + v * 3 + 2] = mesh.nodes[ni + 2];
			}
		}
		const norm64 = new Float64Array(ntri * 9);
		for (let t = 0; t < ntri; t++) {
			const i = t * 9;
			const ax = pos64[i + 0], ay = pos64[i + 1], az = pos64[i + 2];
			const bx = pos64[i + 3], by = pos64[i + 4], bz = pos64[i + 5];
			const cx = pos64[i + 6], cy = pos64[i + 7], cz = pos64[i + 8];
			const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
			const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
			let nx = e1y * e2z - e1z * e2y;
			let ny = e1z * e2x - e1x * e2z;
			let nz = e1x * e2y - e1y * e2x;
			const l = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1;
			nx /= l; ny /= l; nz /= l;
			// Snap any axis-aligned normal back to its exact value — kills any
			// residual sub-bit FP noise that would dapple coplanar shading.
			if (Math.abs(nx) > 0.9999) { nx = Math.sign(nx); ny = 0; nz = 0; }
			else if (Math.abs(ny) > 0.9999) { ny = Math.sign(ny); nx = 0; nz = 0; }
			else if (Math.abs(nz) > 0.9999) { nz = Math.sign(nz); nx = 0; ny = 0; }
			for (let k = 0; k < 3; k++) {
				norm64[i + k * 3 + 0] = nx;
				norm64[i + k * 3 + 1] = ny;
				norm64[i + k * 3 + 2] = nz;
			}
		}
		// Push dielectric volume hulls slightly back so coplanar conductor
		// plates win the depth test cleanly. In field mode we color all
		// surfaces by |E| anyway — z-fighting isn't a concern.
		const offset: [number, number] | undefined =
			kind === 'dielectric' && !field_norm ? [2, 2] : undefined;
		// Per-vertex scalar lookup from the global per-node field array
		let scalars: Float32Array | undefined;
		if (field_norm) {
			scalars = new Float32Array(ntri * 3);
			for (let t = 0; t < ntri; t++) {
				for (let v = 0; v < 3; v++) {
					scalars[t * 3 + v] = field_norm[idx[t * 3 + v]];
				}
			}
		}
		addMesh(
			gl_state,
			Float32Array.from(pos64),
			Float32Array.from(norm64),
			color_for(kind, name),
			tag,
			offset,
			scalars
		);
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

	// Coalesce renders onto a single rAF tick. Pointer events, the
	// depth-sort worker callback, $effects and the ResizeObserver can all
	// fire several times before the next display refresh — without this
	// they each trigger a full render, e.g. orbiting drove TWO renders per
	// move (one on pointermove, one on the sort result). One render/frame.
	let render_scheduled = false;
	function schedule_render() {
		if (render_scheduled) return;
		render_scheduled = true;
		requestAnimationFrame(() => {
			render_scheduled = false;
			render_frame();
		});
	}

	// ── Pointer / wheel handlers (orbit/pan/zoom analog rapidpassives) ──
	function on_wheel(e: WheelEvent) {
		e.preventDefault();
		const factor = e.deltaY > 0 ? 1.1 : 1 / 1.1;
		camera = { ...camera, distance: camera.distance * factor };
		schedule_render();
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
		schedule_render();
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

		const ro = new ResizeObserver(() => mounted && schedule_render());
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

	// React to mesh / toggles / field / density changes
	$effect(() => {
		mesh; wireframe; show_geometry; show_wireframe; show_field; field; point_density;
		td_trajectory;
		if (!mounted || !gl_state) return;
		needs_rebuild = true;
		schedule_render();
	});


	// Refit camera when the visible payload changes (mesh, wireframe or
	// a time-domain field trajectory).
	$effect(() => {
		if (!mounted) return;
		if (td_trajectory) camera = fitCamera(td_trajectory.bbox.min, td_trajectory.bbox.max);
		else if (mesh) camera = fitCamera(mesh.bbox.min, mesh.bbox.max);
		else if (wireframe) camera = fitCamera(wireframe.bbox.min, wireframe.bbox.max);
	});

	// Upload mesh to the worker's viz cache once per mesh change. The worker
	// then holds the nodes+tets+CDF and `viz_sample` only does the cheap
	// random-sampling pass per density tick.
	let viz_mesh_ready_for: MeshData | null = $state(null);
	$effect(() => {
		const m = mesh;
		// Whenever the mesh changes (file load, regen), drop the old GPU splat
		// cloud immediately. Otherwise the previous file's field samples linger
		// in their old coordinates on top of the new geometry until the
		// sampler finishes resampling.
		if (gl_state) {
			setPointCloud(gl_state, EMPTY_F32, EMPTY_F32);
			field_range = null;
		}
		if (!m) { viz_mesh_ready_for = null; return; }
		viz_mesh_ready_for = null;
		viz_load_mesh(m).then(() => { viz_mesh_ready_for = m; }).catch((e) => console.error('viz_load_mesh', e));
	});

	// Async point-cloud sampling: re-runs whenever `show_field`, `field`, or
	// `point_density` change. Old samples are replaced atomically when the
	// worker returns. A monotonically-increasing token guards against
	// out-of-order responses (e.g. user drags slider faster than the worker
	// can answer).
	let viz_sample_token = 0;
	$effect(() => {
		const ready = viz_mesh_ready_for;
		const f = field;
		const dens = point_density;
		const want = show_field;
		if (!gl_state || !ready || !want || !f) return;
		const total_pts = Math.max(500, Math.round(dens * 50000));
		const my_token = ++viz_sample_token;
		viz_sample(f, total_pts).then((r) => {
			if (my_token !== viz_sample_token || !gl_state) return;
			field_range = r.field_range;
			last_range = r;
			apply_scale_mode(gl_state, scale_mode, r);
			setPointCloud(gl_state, r.positions, r.abc);
			schedule_render();
		}).catch((e) => console.error('viz_sample', e));
	});

	// Reapply colormap range without resampling when the user flips Lin/Log.
	let last_range: { log_floor: number; log_range: number; field_range: { min: number; max: number } } | null = null;
	function apply_scale_mode(
		gl: GLState,
		mode: 'log' | 'lin',
		r: { log_floor: number; log_range: number; field_range: { min: number; max: number } },
	) {
		setPointScaleMode(gl, mode);
		if (mode === 'log') setPointRange(gl, r.log_floor, r.log_range);
		else setPointRange(gl, r.field_range.min, r.field_range.max - r.field_range.min);
	}
	$effect(() => {
		const mode = scale_mode;
		if (!gl_state || !last_range) return;
		apply_scale_mode(gl_state, mode, last_range);
		schedule_render();
	});

	// Wave animation: while `show_field` is on AND the `animate_field` prop is
	// true, drive the shader's phase uniform at 2π·anim_speed·t. (Real ω is
	// way too fast for 60 fps — we show a slowed-down phase rotation.)
	let anim_raf: number | null = null;
	$effect(() => {
		const want = show_field && animate_field;
		if (anim_raf != null) { cancelAnimationFrame(anim_raf); anim_raf = null; }
		if (!want || !gl_state) {
			if (gl_state) { setPointPhase(gl_state, 0); schedule_render(); }
			return;
		}
		const t0 = performance.now();
		const tick = () => {
			if (!gl_state) return;
			const t = (performance.now() - t0) * 0.001;
			setPointPhase(gl_state, t * 2 * Math.PI * anim_speed);
			schedule_render();
			anim_raf = requestAnimationFrame(tick);
		};
		anim_raf = requestAnimationFrame(tick);
	});

	// ── Time-domain field animation ────────────────────────────────────
	// `td_trajectory` carries an energy-weighted point cloud (sampled the
	// same way as the frequency-domain field viz) plus a per-frame |E|/|H|
	// magnitude. The notebook page owns the time slider / play loop and
	// feeds the frame index through `td_frame`; this just renders it.
	// A trajectory is "in TD field mode" only while the Field toggle is on —
	// the cloud, its colourbar and its channel toolbar all ride that switch.
	const in_td_mode = $derived(td_trajectory != null && show_field);
	let td_positions: Float32Array | null = $state(null);

	$effect(() => {
		const traj = td_trajectory;
		const want = show_field;
		// No trajectory, or the Field toggle is off: drop the cloud.
		if (gl_state && (!traj || !want)) setPointCloud(gl_state, EMPTY_F32, EMPTY_F32);
		if (!traj || !want) {
			td_positions = null;
			if (traj) needs_rebuild = true;
			schedule_render();
			return;
		}
		td_positions = Float32Array.from(traj.points);
		needs_rebuild = true;
		if (mounted) camera = fitCamera(traj.bbox.min, traj.bbox.max);
		schedule_render();
	});

	// Upload the current frame's magnitude as a static-scalar point cloud.
	$effect(() => {
		const traj = td_trajectory;
		const frame = td_frame;
		const ch = td_channel;
		const pos = td_positions;
		const mode = scale_mode;
		if (!gl_state || !traj || !pos) return;
		const abc = td_abc_for_frame(traj, frame, ch);
		const fmax = ch === 'H' ? traj.field_max.H : traj.field_max.E;
		setPointScaleMode(gl_state, mode);
		if (mode === 'log') {
			// floor / span are (log10(min), decades) in log mode — a fixed
			// decade window below the per-channel peak.
			setPointRange(
				gl_state,
				Math.log10(Math.max(fmax, 1e-30)) - TD_LOG_DECADES,
				TD_LOG_DECADES,
			);
		} else {
			setPointRange(gl_state, 0, fmax);
		}
		setPointCloud(gl_state, pos, abc);
		schedule_render();
	});

	// Colourbar range for the time-domain cloud — a fixed 0…max scale held
	// constant across the whole animation so frames are comparable.
	const td_field_range = $derived(
		td_trajectory
			? {
					min: 0,
					max: td_channel === 'H'
						? td_trajectory.field_max.H
						: td_trajectory.field_max.E,
					decades: scale_mode === 'log' ? TD_LOG_DECADES : 0,
				}
			: null,
	);

	function fmt_eng(v: number): string {
		if (!isFinite(v) || v <= 0) return '0';
		const exp = Math.floor(Math.log10(v) / 3) * 3;
		const m = v / Math.pow(10, exp);
		const prefix = ({ '-12': 'p', '-9': 'n', '-6': 'µ', '-3': 'm', '0': '', '3': 'k', '6': 'M', '9': 'G' } as Record<string, string>)[String(exp)];
		const mantissa = m >= 100 ? m.toFixed(0) : m >= 10 ? m.toFixed(1) : m.toFixed(2);
		return prefix !== undefined ? `${mantissa} ${prefix}` : `${m.toFixed(1)}e${exp}`;
	}

	// The colourbar tracks the frequency-domain field range, or the
	// time-domain trajectory range when a TD animation is shown.
	const active_range = $derived(td_field_range ?? field_range);

	// Colorbar ticks. Log mode: one per decade. Lin mode: 5 evenly-spaced.
	// The time-domain cloud is always linear.
	const colorbar_ticks = $derived.by(() => {
		const fr = active_range;
		if (!fr) return [] as { frac: number; label: string }[];
		const out: { frac: number; label: string }[] = [];
		if (scale_mode === 'log') {
			const log_max = Math.log10(fr.max);
			const log_min = log_max - Math.max(fr.decades, 0.5);
			const n_dec = Math.max(1, Math.round(fr.decades));
			for (let i = 0; i <= n_dec; i++) {
				const v = Math.pow(10, log_min + (log_max - log_min) * (i / n_dec));
				out.push({ frac: i / n_dec, label: fmt_eng(v) });
			}
		} else {
			const n = 4;
			for (let i = 0; i <= n; i++) {
				const v = fr.min + (fr.max - fr.min) * (i / n);
				out.push({ frac: i / n, label: fmt_eng(v) });
			}
		}
		return out;
	});

	const tag_legend = $derived.by(() => {
		// Wireframe mode: emit one legend item per OCC entity.
		if (!mesh && wireframe) {
			const items: { name: string; color: string; kind: Kind; rank: number; tag: number }[] = [];
			for (const e of wireframe.entities) {
				const k = classify(e.name) ?? 'conductor';
				const c = e.color;
				items.push({
					name: e.name,
					color: `rgb(${(c[0] * 255) | 0},${(c[1] * 255) | 0},${(c[2] * 255) | 0})`,
					kind: k, rank: k === 'conductor' ? 0 : k === 'port' ? 1 : k === 'gnd' ? 2 : 3,
					tag: e.tag,
				});
			}
			return items;
		}
		if (!mesh) return [] as { name: string; color: string; kind: Kind; rank: number; tag: number }[];
		const seen = new Set<number>();
		const items: { name: string; color: string; kind: Kind; rank: number; tag: number }[] = [];
		const add = (tag: number, kind: Kind) => {
			if (seen.has(tag)) return;
			seen.add(tag);
			const name = mesh!.phys_names.get(tag) ?? '';
			if (!name) return;
			// ABC is rendered transparently; suppress from the legend too.
			if (classify(name) === null) return;
			const c = color_for(kind, name);
			const rank = kind === 'conductor' ? 0 : kind === 'port' ? 1 : kind === 'gnd' ? 2 : 3;
			items.push({
				name: pretty_label(name),
				color: `rgb(${(c[0] * 255) | 0},${(c[1] * 255) | 0},${(c[2] * 255) | 0})`,
				kind, rank, tag
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
			const k = classify(name);
			if (k) add(tag, k);
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

	<div class="overlay-stack">
		{#if tag_legend.length > 0 && show_geometry}
			<div class="overlay-panel">
				<div class="op-title">Geometry</div>
				<div class="op-body">
					{#each tag_legend as l}
						<button
							class="legend-item"
							class:hidden={hidden_tags.has(l.tag)}
							onclick={() => toggle_tag(l.tag)}
							title="Click to toggle"
						>
							<span class="swatch" style="background: {l.color};"></span>
							<span class="legend-name">{l.name}</span>
						</button>
					{/each}
				</div>
			</div>
		{/if}

		{#if (show_field && field_range) || (in_td_mode && td_field_range)}
			<div class="overlay-panel cb-panel">
				<div class="op-title">
					{CHANNEL_META[in_td_mode ? td_channel : field_channel].sym} ·
					{CHANNEL_META[in_td_mode ? td_channel : field_channel].unit}
					<span class="cb-mode">({scale_mode})</span>
				</div>
				<div class="cb-body">
					<div class="cb-gradient">
						{#each colorbar_ticks as tk}
							<span class="cb-tick" style="bottom: {tk.frac * 100}%"></span>
						{/each}
					</div>
					<div class="cb-labels">
						{#each colorbar_ticks as tk}
							<span class="cb-label" style="bottom: {tk.frac * 100}%">{tk.label}</span>
						{/each}
					</div>
				</div>
			</div>
		{/if}
	</div>

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
		<button class="tb" onclick={save_png}>
			<span class="tip">Save PNG<kbd>Ctrl+S</kbd></span>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5">
				<path d="M2 10v3h12v-3" /><path d="M8 2v8" /><path d="M5 7l3 3 3-3" />
			</svg>
		</button>
		{#if show_field}
			<span class="tb-sep" aria-hidden="true"></span>
			{#each (['E', 'J', 'H'] as const) as ch}
				{@const enabled = available_channels.includes(ch)}
				<button
					class="tb tb-label"
					class:active={field_channel === ch}
					disabled={!enabled}
					onclick={() => { if (enabled) field_channel = ch; }}
				>
					<span class="tip">{ch === 'E' ? 'E-field (V/m)' :
					                   ch === 'J' ? 'Current density σE (A/m²)' :
					                                'Magnetic field ∇×E/(jωμ) (A/m)'}</span>
					{ch}
				</button>
			{/each}
			<button
				class="tb tb-label tb-scale"
				onclick={() => (scale_mode = scale_mode === 'log' ? 'lin' : 'log')}
			>
				<span class="tip">{scale_mode === 'log' ? 'Switch to linear scale' : 'Switch to log scale'}</span>
				{scale_mode}
			</button>
		{/if}
		{#if in_td_mode}
			<span class="tb-sep" aria-hidden="true"></span>
			{#each (['E', 'H'] as const) as ch}
				<button
					class="tb tb-label"
					class:active={td_channel === ch}
					onclick={() => (td_channel = ch)}
				>
					<span class="tip">{ch === 'E' ? 'E-field magnitude' : 'H-field magnitude'}</span>
					{ch}
				</button>
			{/each}
			<button
				class="tb tb-label tb-scale"
				onclick={() => (scale_mode = scale_mode === 'log' ? 'lin' : 'log')}
			>
				<span class="tip">{scale_mode === 'log' ? 'Switch to linear scale' : 'Switch to log scale'}</span>
				{scale_mode}
			</button>
		{/if}
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

	.overlay-stack {
		position: absolute;
		top: 10px;
		left: 10px;
		display: flex;
		flex-direction: column;
		gap: 6px;
		max-height: calc(100% - 20px);
	}
	.overlay-panel {
		background: var(--bg-surface);
		border: 1px solid var(--border-subtle);
		padding: 8px 10px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: 96px;
	}
	.op-title {
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 1.5px;
		color: var(--accent);
		font-weight: 600;
	}
	.op-body {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}
	.legend-item {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 3px 4px;
		margin: 0 -4px;
		background: transparent;
		border: 0;
		color: var(--text-muted);
		cursor: pointer;
		text-align: left;
		font-family: inherit;
		font-size: inherit;
		text-transform: none;
		letter-spacing: 0;
		transition: background var(--transition), color var(--transition);
	}
	.legend-item:hover { background: var(--accent-dim); color: var(--text); }
	.legend-item.hidden { color: var(--text-dim); }
	.legend-item.hidden .swatch { opacity: 0.25; }
	.legend-item.hidden .legend-name { text-decoration: line-through; }
	.swatch { width: 10px; height: 10px; flex-shrink: 0; transition: opacity var(--transition); }

	.viewer-toolbar {
		position: absolute;
		top: 10px;
		right: 10px;
		z-index: 10;
		display: flex;
		flex-wrap: wrap;
		justify-content: flex-end;
		gap: 2px;
		max-width: calc(100% - 20px);
	}
	.tb-sep {
		display: inline-block;
		width: 1px;
		height: 20px;
		margin: 4px 4px;
		background: var(--border);
	}
	.tb.tb-label {
		font-family: var(--font-mono);
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.tb.tb-label.active {
		color: var(--accent);
		background: var(--accent-dim);
		border-color: var(--accent);
	}
	.tb.tb-label:disabled {
		color: var(--text-dim);
		cursor: default;
		opacity: 0.4;
		border-color: var(--border);
		background: var(--bg-surface);
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

	.cb-panel {
		padding: 12px 14px;
		gap: 14px;       /* extra breathing room between title and gradient */
	}
	.cb-body {
		display: flex;
		flex-direction: row;
		gap: 10px;
		align-items: stretch;
		height: 180px;
		position: relative;
	}
	.cb-gradient {
		width: 14px;
		flex-shrink: 0;
		position: relative;
		background: linear-gradient(
			to top,
			#000004 0%,
			#1B0C42 14%,
			#420A68 28%,
			#6A176E 43%,
			#932667 57%,
			#BB3754 71%,
			#DD513A 85%,
			#FCFFA4 100%
		);
		border: 1px solid var(--text-dim);
	}
	.cb-tick {
		position: absolute;
		right: -5px;
		width: 5px;
		height: 1px;
		background: var(--text-muted);
		transform: translateY(50%);
	}
	.cb-labels {
		position: relative;
		flex: 1;
		min-width: 36px;
	}
	.cb-label {
		position: absolute;
		left: 4px;
		transform: translateY(50%);
		font-size: var(--fs-xs);
		line-height: 1;
		color: var(--text-muted);
		white-space: nowrap;
	}

</style>
