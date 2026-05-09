<script lang="ts">
	import { onMount } from 'svelte';
	import {
		initGL, disposeGL, clearMeshes, addMesh, addLineMesh, setBBox,
		render3D, fitCamera, setTagVisible, setClipPlane,
		type GLState, type Camera
	} from '$lib/render/canvas3d';
	import type { MeshData } from '$lib/msh';
	import { palette } from '$lib/theme';

	let {
		mesh = null,
		mode = 'geometry',
		field = null
	}: {
		mesh?: MeshData | null;
		mode?: 'geometry' | 'mesh' | 'both' | 'field';
		/** Per-node scalar (e.g. |E| in V/m). Length must match mesh.n_nodes.
		 *  Activates field-colormap rendering when set. */
		field?: Float32Array | null;
	} = $props();

	let canvas = $state<HTMLCanvasElement | null>(null);
	let container = $state<HTMLDivElement | null>(null);
	let gl_state: GLState | null = null;
	let camera: Camera = { theta: Math.PI / 4, phi: Math.PI / 4, distance: 1, target: [0, 0, 0] };
	let z_flip = 1;
	let mounted = false;
	let needs_rebuild = true;
	let cursor_world = $state({ x: 0, y: 0 });
	let visible_tags = $state(new Set<number>());
	let iso_level = $state(0.5);          // 0..1 normalized — field-magnitude iso-surface level
	let iso_enabled = $state(false);

	function toggle_tag(tag: number) {
		if (!gl_state) return;
		const next = new Set(visible_tags);
		if (next.has(tag)) next.delete(tag); else next.add(tag);
		visible_tags = next;
		setTagVisible(gl_state, tag, next.has(tag));
		render_frame();
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
			render_frame();
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
		if (!mesh) return;
		animate_camera(fitCamera(mesh.bbox.min, mesh.bbox.max), 350);
	}
	export function rotate_90() {
		const base = effective_camera();
		animate_camera({ ...base, target: [...base.target] as [number, number, number], theta: base.theta + Math.PI / 2 }, 400);
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
		if (!gl_state || !mesh) return;
		clearMeshes(gl_state);

		const showFaces = mode === 'geometry' || mode === 'both' || mode === 'field';
		const showWire = mode === 'mesh' || mode === 'both';
		const useField = mode === 'field' && field != null;
		// Compute log-scaled field per node (|E| spans many orders of magnitude
		// across a passive RFIC structure).
		let f_norm: Float32Array | null = null;
		if (useField && field) {
			let max_v = 0;
			for (let i = 0; i < field.length; i++) {
				if (field[i] > max_v) max_v = field[i];
			}
			f_norm = new Float32Array(field.length);
			if (max_v > 0) {
				const log_max = Math.log10(max_v + 1e-30);
				const log_floor = log_max - 4; // 4 decades below max
				for (let i = 0; i < field.length; i++) {
					const v = field[i];
					if (v <= 0) f_norm[i] = 0;
					else {
						const l = Math.log10(v);
						f_norm[i] = Math.max(0, Math.min(1, (l - log_floor) / (log_max - log_floor)));
					}
				}
			}
		}

		setBBox(gl_state, mesh.bbox.min, mesh.bbox.max);
		field_norm = f_norm;   // visible to push_group below

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
				push_group(idx, kind, name, tag);
			}
			// 2) Implicit volume hulls
			const vol_b = build_volume_boundaries(mesh);
			for (const [vtag, idx] of vol_b.entries()) {
				const name = mesh.phys_names.get(vtag) ?? '';
				if (!name || name.startsWith('_mat_')) continue;
				push_group(idx, 'dielectric', name, vtag);
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
			addLineMesh(gl_state, Float32Array.from(edges), hex(palette.textDim), -1);
		}

		// Iso-surface (volumetric field viz) — only in field mode when enabled.
		// Uses the LOG-mapped normalized field so the slider 0..1 corresponds
		// linearly to log(|E|) decades rather than to a raw V/m value range
		// (which would otherwise be unusably skewed toward the very high end).
		if (useField && f_norm && iso_enabled) {
			const iso = build_iso_surface(iso_level, f_norm);
			if (iso.positions.length > 0) {
				const scalars = new Float32Array(iso.positions.length / 3);
				scalars.fill(iso_level);
				addMesh(gl_state, iso.positions, iso.normals, [1, 1, 1], -2, undefined, scalars);
			}
		}

		// Reset visibility tracking after rebuild — everything visible by default
		const tags = new Set<number>();
		for (const m of gl_state.meshes) tags.add(m.tag);
		visible_tags = tags;

		needs_rebuild = false;
	}

	/** Marching-tets iso-surface extraction. For each tet: classify each
	 *  vertex above/below the threshold, then emit 0/1/2 triangles where
	 *  edges cross the iso-level. Returns positions+normals as f32 arrays. */
	function build_iso_surface(level: number, f: Float32Array): { positions: Float32Array; normals: Float32Array } {
		if (!mesh) return { positions: new Float32Array(0), normals: new Float32Array(0) };
		const nodes = mesh.nodes;
		const tets = mesh.tets;
		const ntets = mesh.tets.length / 4;
		const tri_pos: number[] = [];
		// Each tet has 6 edges, each edge = pair of tet-local indices
		const tet_edges: [number, number][] = [
			[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
		];
		const interp = (a: number, b: number, fa: number, fb: number): [number, number, number] => {
			let t = fa === fb ? 0.5 : (level - fa) / (fb - fa);
			t = Math.max(0, Math.min(1, t));
			return [
				nodes[a * 3 + 0] * (1 - t) + nodes[b * 3 + 0] * t,
				nodes[a * 3 + 1] * (1 - t) + nodes[b * 3 + 1] * t,
				nodes[a * 3 + 2] * (1 - t) + nodes[b * 3 + 2] * t
			];
		};
		for (let ti = 0; ti < ntets; ti++) {
			const v = [tets[ti * 4], tets[ti * 4 + 1], tets[ti * 4 + 2], tets[ti * 4 + 3]];
			const fv = [f[v[0]], f[v[1]], f[v[2]], f[v[3]]];
			let mask = 0;
			for (let i = 0; i < 4; i++) if (fv[i] >= level) mask |= 1 << i;
			if (mask === 0 || mask === 15) continue;
			// Find crossing edges (edge endpoints have different above/below)
			const cross_pts: [number, number, number][] = [];
			for (const [i, j] of tet_edges) {
				const ai = (mask >> i) & 1;
				const aj = (mask >> j) & 1;
				if (ai !== aj) cross_pts.push(interp(v[i], v[j], fv[i], fv[j]));
			}
			if (cross_pts.length === 3) {
				for (const p of cross_pts) tri_pos.push(p[0], p[1], p[2]);
			} else if (cross_pts.length === 4) {
				// Quad → split into 2 tris (simple fan; ordering of cross_pts is
				// already coherent because tet_edges enumerates in fixed order)
				const [a, b, c, d] = cross_pts;
				tri_pos.push(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
				tri_pos.push(a[0], a[1], a[2], c[0], c[1], c[2], d[0], d[1], d[2]);
			}
		}
		const positions = Float32Array.from(tri_pos);
		// Face normals via cross product (no smoothing — clean per-face shading)
		const ntri = positions.length / 9;
		const normals = new Float32Array(positions.length);
		for (let t = 0; t < ntri; t++) {
			const i = t * 9;
			const ax = positions[i], ay = positions[i + 1], az = positions[i + 2];
			const bx = positions[i + 3], by = positions[i + 4], bz = positions[i + 5];
			const cx = positions[i + 6], cy = positions[i + 7], cz = positions[i + 8];
			const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
			const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
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
		return { positions, normals };
	}

	let field_norm: Float32Array | null = null;
	function push_group(idx: number[], kind: Kind, name: string, tag: number) {
		if (!gl_state || !mesh) return;
		const ntri = idx.length / 3;
		if (ntri === 0) return;

		// Stage everything in f64 so cross-product normals on coplanar
		// triangles are bit-identical. Only quantize to f32 at GPU upload.
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

	// React to mesh / mode / field changes
	$effect(() => {
		mesh; mode; field;
		if (!mounted || !gl_state) return;
		needs_rebuild = true;
		render_frame();
	});

	// React to iso-level changes (full rebuild — recomputes the iso-surface)
	$effect(() => {
		iso_level; iso_enabled;
		if (!mounted || !gl_state) return;
		needs_rebuild = true;
		render_frame();
	});

	// Refit camera only when mesh changes
	$effect(() => {
		if (mesh && mounted) camera = fitCamera(mesh.bbox.min, mesh.bbox.max);
	});

	const tag_legend = $derived.by(() => {
		if (!mesh) return [] as { name: string; color: string; kind: Kind; rank: number; tag: number }[];
		const seen = new Set<number>();
		const items: { name: string; color: string; kind: Kind; rank: number; tag: number }[] = [];
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
				<button
					class="legend-item"
					class:hidden={!visible_tags.has(l.tag)}
					onclick={() => toggle_tag(l.tag)}
					title="Click to toggle"
				>
					<span class="swatch" style="background: {l.color};"></span>
					<span class="legend-name">{l.name}</span>
				</button>
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

	{#if mode === 'field' && field}
		<div class="iso-control">
			<label>
				<input
					type="checkbox"
					bind:checked={iso_enabled}
				/>
				iso-surface
			</label>
			{#if iso_enabled}
				<input
					class="iso-slider"
					type="range"
					min={0.05}
					max={0.95}
					step={0.01}
					bind:value={iso_level}
				/>
				<span class="iso-readout">|E| · {(iso_level * 100).toFixed(0)}%</span>
			{/if}
		</div>
	{/if}
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
		padding: 4px;
		display: flex;
		flex-direction: column;
		gap: 1px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.legend-item {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 3px 6px;
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

	.iso-control {
		position: absolute;
		bottom: 8px;
		right: 10px;
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 4px 8px;
		background: rgba(24, 24, 29, 0.75);
		border: 1px solid var(--border-subtle);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
	}
	.iso-control label {
		display: flex;
		align-items: center;
		gap: 4px;
		cursor: pointer;
	}
	.iso-control input[type='checkbox'] {
		accent-color: var(--accent);
		cursor: pointer;
	}
	.iso-slider {
		appearance: none;
		width: 160px;
		height: 2px;
		background: var(--input-border);
		outline: none;
	}
	.iso-slider::-webkit-slider-thumb {
		appearance: none;
		width: 10px;
		height: 14px;
		background: var(--accent);
		cursor: pointer;
	}
	.iso-slider::-moz-range-thumb {
		width: 10px;
		height: 14px;
		background: var(--accent);
		border: 0;
		cursor: pointer;
	}
	.iso-readout {
		min-width: 64px;
		text-align: right;
		color: var(--accent);
	}
</style>
