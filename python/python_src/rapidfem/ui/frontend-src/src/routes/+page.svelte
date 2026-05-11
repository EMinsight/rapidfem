<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import {
		runCode, meshGeometry, solve, readFile, writeFile,
		subscribeBus, meshPayloadToMeshData, sparamsToSMatrices, health,
		type RunResponse, type MeshResponse, type SolveResponse, type SMatrix, type BusEvent,
	} from '$lib/api';
	import type { MeshData } from '$lib/msh';
	import MeshViewer from '$lib/components/MeshViewer.svelte';
	import ResultsPanel from '$lib/components/ResultsPanel.svelte';
	import CodeEditor from '$lib/components/CodeEditor.svelte';
	import FileBrowser from '$lib/components/FileBrowser.svelte';
	import Resizer from '$lib/components/Resizer.svelte';

	let status = $state('idle');
	let workdir = $state('');
	let active_path = $state<string | null>(null);
	let code = $state('');
	let dirty = $state(false);
	let log_lines = $state<string[]>([]);

	let mesh_data = $state<MeshData | null>(null);
	let smats = $state<SMatrix[]>([]);
	let freqs = $state<number[]>([]);
	let fields_raw = $state<(number[] | null)[][] | null>(null);
	let field_freq_idx = $state(0);
	let field_port_idx = $state(0);
	let field_abc = $derived<Float32Array | null>(
		fields_raw && fields_raw[field_freq_idx] && fields_raw[field_freq_idx][field_port_idx]
			? new Float32Array(fields_raw[field_freq_idx][field_port_idx] as number[])
			: null,
	);

	let geom_busy = $state(false);
	let mesh_busy = $state(false);
	let solve_busy = $state(false);
	let last_geom_stats = $state<{ n_entities: number; n_triangles: number } | null>(null);
	let last_mesh_stats = $state<{ n_tets: number; n_tris: number; mesh_time_s: number } | null>(null);
	let last_solve_stats = $state<{ n_freq: number; n_dofs: number; solve_time_s: number } | null>(null);

	let display = $state<'view3d' | 'plots'>('view3d');
	let show_geometry = $state(true);
	let show_wireframe = $state(false);
	let show_field = $state(false);
	let viewer: ReturnType<typeof MeshViewer> | undefined = $state();
	let unsub_bus: (() => void) | null = null;
	let geom_debounce: ReturnType<typeof setTimeout> | null = null;

	function on_keydown(e: KeyboardEvent) {
		// Skip when typing in editor / inputs.
		const tag = (e.target as HTMLElement | null)?.tagName;
		if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
		if ((e.target as HTMLElement | null)?.closest?.('.cm-editor')) return;
		if (display !== 'view3d') return;
		switch (e.key) {
			case 'f': case 'F': viewer?.fit_view(); e.preventDefault(); break;
			case '+': case '=': viewer?.zoom_in(); e.preventDefault(); break;
			case '-': case '_': viewer?.zoom_out(); e.preventDefault(); break;
			case 'r': case 'R':
				if (!e.ctrlKey && !e.metaKey) { viewer?.rotate_90(); e.preventDefault(); }
				break;
			case 'z': case 'Z':
				if (!e.ctrlKey && !e.metaKey) { viewer?.flip_z(); e.preventDefault(); }
				break;
			case 'g': case 'G': show_geometry = !show_geometry; e.preventDefault(); break;
			case 'm': case 'M': show_wireframe = !show_wireframe; e.preventDefault(); break;
			case 'e': case 'E':
				if (last_solve_stats) { show_field = !show_field; e.preventDefault(); }
				break;
		}
	}

	// ── Layout: pane widths + collapse state ─────────────────────────────────
	const COLLAPSED_W = 32;
	let files_w = $state(220);
	let editor_w = $state(0);  // computed once on mount as ratio of remaining
	let files_collapsed = $state(false);
	let editor_collapsed = $state(false);
	let viewer_collapsed = $state(false);
	let main_el: HTMLElement | undefined = $state();
	let editor_pane_el: HTMLElement | undefined = $state();

	let output_h = $state(140);
	let output_collapsed = $state(false);

	function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

	function load_layout() {
		try {
			const raw = localStorage.getItem('rapidfem.layout');
			if (!raw) return;
			const s = JSON.parse(raw);
			if (typeof s.files_w === 'number') files_w = s.files_w;
			if (typeof s.editor_w === 'number') editor_w = s.editor_w;
			if (typeof s.files_collapsed === 'boolean') files_collapsed = s.files_collapsed;
			if (typeof s.editor_collapsed === 'boolean') editor_collapsed = s.editor_collapsed;
			if (typeof s.viewer_collapsed === 'boolean') viewer_collapsed = s.viewer_collapsed;
			if (typeof s.output_h === 'number') output_h = s.output_h;
			if (typeof s.output_collapsed === 'boolean') output_collapsed = s.output_collapsed;
		} catch {}
	}
	function save_layout() {
		try {
			localStorage.setItem('rapidfem.layout', JSON.stringify({
				files_w, editor_w, files_collapsed, editor_collapsed, viewer_collapsed,
				output_h, output_collapsed,
			}));
		} catch {}
	}

	// Drag-to-collapse: per-axis tracker accumulates raw drag (unclamped).
	// On drag start the tracker is initialized from the current rendered
	// width so the first move continues smoothly. The tracker is reset
	// defensively at the start of every drag so stale values from earlier
	// drags can never sabotage threshold detection.
	const COLLAPSE_AT = 80;
	const EXPAND_AT  = 50;
	let files_track = 220;
	let editor_track = 0;
	let output_track = 140;

	function on_files_drag_start() {
		files_track = files_collapsed ? 0 : files_w;
	}
	function on_files_resize(dx: number) {
		files_track += dx;
		if (files_collapsed) {
			if (files_track > EXPAND_AT) {
				files_collapsed = false;
				files_w = Math.max(files_w, 220);
				files_track = files_w;
				save_layout();
			}
			return;
		}
		if (files_track < COLLAPSE_AT) {
			files_collapsed = true;
			files_track = 0;
			save_layout();
			return;
		}
		files_w = clamp(files_track, 140, 480);
		save_layout();
	}

	function avail_for_editor_viewer(): number {
		if (!main_el) return 0;
		const filesPx = files_collapsed ? COLLAPSED_W : files_w;
		return main_el.clientWidth - filesPx - 8;
	}

	function on_editor_drag_start() {
		const avail = avail_for_editor_viewer();
		if (editor_collapsed)       editor_track = COLLAPSED_W;
		else if (viewer_collapsed)  editor_track = Math.max(avail - COLLAPSED_W, 240);
		else                         editor_track = editor_w;
	}
	function on_editor_resize(dx: number) {
		if (!main_el) return;
		editor_track += dx;
		const totalAvail = avail_for_editor_viewer();

		if (editor_collapsed) {
			if (editor_track > COLLAPSED_W + EXPAND_AT) {
				editor_collapsed = false;
				editor_w = Math.max(editor_w, 320);
				editor_track = editor_w;
				save_layout();
			}
			return;
		}
		if (viewer_collapsed) {
			// editor occupies (totalAvail - COLLAPSED_W); drag left to re-expand viewer.
			if (editor_track < totalAvail - COLLAPSED_W - EXPAND_AT) {
				viewer_collapsed = false;
				editor_w = clamp(editor_track, 240, totalAvail - 240);
				editor_track = editor_w;
				save_layout();
			}
			return;
		}

		const viewerNext = totalAvail - editor_track;
		if (editor_track < COLLAPSE_AT) {
			editor_collapsed = true;
			editor_track = COLLAPSED_W;
			save_layout();
			return;
		}
		if (viewerNext < COLLAPSE_AT) {
			viewer_collapsed = true;
			editor_w = clamp(editor_track, 240, totalAvail - COLLAPSED_W);
			editor_track = totalAvail - COLLAPSED_W;
			save_layout();
			return;
		}
		editor_w = clamp(editor_track, 240, totalAvail - 240);
		save_layout();
	}

	function on_output_drag_start() {
		output_track = output_collapsed ? 24 : output_h;
	}
	function on_output_resize(dy: number) {
		if (!editor_pane_el) return;
		// dy positive = mouse moves down = output shrinks.
		output_track -= dy;
		if (output_collapsed) {
			if (output_track > 24 + EXPAND_AT) {
				output_collapsed = false;
				output_h = Math.max(output_h, 140);
				output_track = output_h;
				save_layout();
			}
			return;
		}
		if (output_track < 40) {
			output_collapsed = true;
			output_track = 24;
			save_layout();
			return;
		}
		const maxH = editor_pane_el.clientHeight - 100;
		output_h = clamp(output_track, 60, maxH);
		save_layout();
	}

	function init_editor_w() {
		if (!main_el) return;
		const filesPx = files_collapsed ? COLLAPSED_W : files_w;
		const avail = main_el.clientWidth - filesPx - 8;  // minus 2 × 4px resizer
		if (editor_w === 0) editor_w = Math.round(avail / 2);
	}

	onMount(async () => {
		load_layout();
		init_editor_w();
		try {
			const h = await health();
			workdir = h.workdir;
		} catch {
			status = 'backend unreachable';
		}
		unsub_bus = subscribeBus((e: BusEvent) => {
			if (e.kind === 'stage_start') status = `${e.stage}…`;
			else if (e.kind === 'stage_end') status = e.ok ? `${e.stage} ok` : `${e.stage} failed`;
		});
		const last = localStorage.getItem('rapidfem.active_path');
		if (last) await open_file(last);
	});

	onDestroy(() => unsub_bus?.());

	async function open_file(path: string) {
		try {
			const content = await readFile(path);
			code = content;
			active_path = path;
			dirty = false;
			localStorage.setItem('rapidfem.active_path', path);
			await run_geometry();
		} catch (e) {
			log_lines = [...log_lines, `[open ${path}] ${e}`];
		}
	}

	async function new_file() {
		const name = prompt('New file name (e.g. patch.py):');
		if (!name) return;
		const path = name.endsWith('.py') ? name : `${name}.py`;
		try {
			await writeFile(path, '# new rapidfem script\nimport rapidfem\n\n');
			await open_file(path);
		} catch (e) {
			log_lines = [...log_lines, `[new ${path}] ${e}`];
		}
	}

	function append_log(label: string, r: RunResponse | MeshResponse | SolveResponse) {
		if (r.stdout) log_lines = [...log_lines, ...r.stdout.split('\n').filter(Boolean).map((l) => `[${label}] ${l}`)];
		if (r.stderr) log_lines = [...log_lines, ...r.stderr.split('\n').filter(Boolean).map((l) => `[${label} err] ${l}`)];
		if (!r.ok && r.error) log_lines = [...log_lines, `[${label}] ${r.error.type}: ${r.error.message}`];
	}

	async function on_save(text: string) {
		code = text;
		dirty = false;
		if (active_path) {
			try {
				await writeFile(active_path, text);
			} catch (e) {
				log_lines = [...log_lines, `[save] ${e}`];
			}
		}
		if (geom_debounce) clearTimeout(geom_debounce);
		geom_debounce = setTimeout(() => {
			geom_debounce = null;
			void run_geometry();
		}, 200);
	}

	function on_change(text: string) {
		if (text !== code) {
			code = text;
			dirty = !!active_path;
		}
	}

	async function run_geometry() {
		if (geom_busy) return;
		geom_busy = true;
		try {
			const r = await runCode(code);
			append_log('run', r);
			if (r.ok) {
				const geo = r.captures.find((c) => c.kind === 'geometry');
				if (geo?.payload) {
					last_geom_stats = {
						n_entities: geo.payload.stats.n_entities,
						n_triangles: geo.payload.stats.n_triangles,
					};
					mesh_data = geometryToMeshData(geo.payload);
				}
			}
		} catch (e) {
			log_lines = [...log_lines, `[run] ${e}`];
		} finally {
			geom_busy = false;
		}
	}

	async function on_mesh() {
		if (mesh_busy) return;
		mesh_busy = true;
		try {
			const r = await meshGeometry(code);
			append_log('mesh', r);
			if (r.ok && r.mesh) {
				mesh_data = meshPayloadToMeshData(r.mesh);
				last_mesh_stats = {
					n_tets: r.mesh.stats.n_tets,
					n_tris: r.mesh.stats.n_tris,
					mesh_time_s: r.mesh.stats.mesh_time_s,
				};
			}
		} catch (e) {
			log_lines = [...log_lines, `[mesh] ${e}`];
		} finally {
			mesh_busy = false;
		}
	}

	async function on_solve() {
		if (solve_busy) return;
		solve_busy = true;
		try {
			const r = await solve(code);
			append_log('solve', r);
			if (r.ok && r.result) {
				freqs = r.result.frequencies;
				smats = sparamsToSMatrices(r.result.sparams);
				fields_raw = r.result.fields ?? null;
				field_freq_idx = 0;
				field_port_idx = 0;
				if (r.mesh) {
					// Replace viewer mesh with the one the solver actually used,
					// so field indices line up with mesh.nodes 1:1.
					mesh_data = meshPayloadToMeshData(r.mesh);
				}
				last_solve_stats = {
					n_freq: r.result.n_freq,
					n_dofs: r.result.n_dofs,
					solve_time_s: r.result.solve_time_s,
				};
			}
		} catch (e) {
			log_lines = [...log_lines, `[solve] ${e}`];
		} finally {
			solve_busy = false;
		}
	}

	function geometryToMeshData(p: import('$lib/api').GeometryPayload): MeshData {
		const phys_names = new Map<number, string>();
		const phys_dim = new Map<number, number>();
		const nodes_flat: number[] = [];
		const tris_flat: number[] = [];
		const tri_phys_flat: number[] = [];
		let next_node = 0;
		p.entities.forEach((ent, i) => {
			const tag = i + 1;
			phys_names.set(tag, ent.name);
			phys_dim.set(tag, ent.dim);
			const n_tri = ent.positions.length / 9;
			for (let t = 0; t < n_tri; t++) {
				for (let v = 0; v < 3; v++) {
					nodes_flat.push(
						ent.positions[t * 9 + v * 3 + 0],
						ent.positions[t * 9 + v * 3 + 1],
						ent.positions[t * 9 + v * 3 + 2],
					);
					tris_flat.push(next_node);
					next_node++;
				}
				tri_phys_flat.push(tag);
			}
		});
		return {
			nodes: new Float64Array(nodes_flat),
			tris: new Uint32Array(tris_flat),
			tri_phys: new Int32Array(tri_phys_flat),
			tets: new Uint32Array(0),
			tet_phys: new Int32Array(0),
			phys_names,
			phys_dim,
			bbox: {
				min: [...p.bbox.min] as [number, number, number],
				max: [...p.bbox.max] as [number, number, number],
			},
		};
	}
</script>

<svelte:head>
	<title>rapidfem</title>
</svelte:head>

<svelte:window onkeydown={on_keydown} />

<div class="app">
	<header>
		<span class="brand">rapidfem</span>
		<span class="nav-sep"></span>
		<span class="workdir">{workdir}</span>
		{#if active_path}
			<span class="path-sep">/</span>
			<span class="active-file">{active_path}{dirty ? ' •' : ''}</span>
		{/if}
		<span class="status">{status}</span>
	</header>

	<main bind:this={main_el}>
		<aside class="pane files-pane" style:flex="0 0 {files_collapsed ? COLLAPSED_W : files_w}px">
			{#if files_collapsed}
				<button
					class="collapsed-strip"
					title="Click or drag to expand"
					aria-label="Expand files"
					onclick={() => { files_collapsed = false; files_w = Math.max(files_w, 220); files_track = files_w; save_layout(); }}
				>
					<span class="strip-label">Files</span>
				</button>
			{:else}
				<div class="pane-inner">
					<FileBrowser
						bind:active_path={active_path}
						onOpen={open_file}
						onNew={new_file}
					/>
				</div>
			{/if}
		</aside>

		<Resizer onStart={on_files_drag_start} onDelta={on_files_resize} />

		<aside class="pane editor-pane" bind:this={editor_pane_el} style:flex={editor_collapsed ? `0 0 ${COLLAPSED_W}px` : (viewer_collapsed ? '1 1 0' : `0 0 ${editor_w}px`)}>
			{#if editor_collapsed}
				<button
					class="collapsed-strip"
					title="Click or drag to expand"
					aria-label="Expand editor"
					onclick={() => { editor_collapsed = false; editor_w = Math.max(editor_w, 320); editor_track = editor_w; save_layout(); }}
				>
					<span class="strip-label">Editor</span>
				</button>
			{:else}
				<div class="pane-inner">
					<div class="toolbar">
						<span class="hint">Ctrl+S → preview</span>
						<button class="primary" onclick={on_mesh} disabled={mesh_busy}>
							{mesh_busy ? 'meshing…' : 'Generate Mesh'}
						</button>
						<button class="primary" onclick={on_solve} disabled={solve_busy}>
							{solve_busy ? 'solving…' : 'Run Simulation'}
						</button>
						<span class="spacer"></span>
						{#if last_solve_stats}
							<span class="stat">{last_solve_stats.n_freq} freq · {last_solve_stats.n_dofs.toLocaleString()} dofs · {last_solve_stats.solve_time_s.toFixed(2)}s</span>
						{:else if last_mesh_stats}
							<span class="stat">{last_mesh_stats.n_tets.toLocaleString()} tets · {last_mesh_stats.mesh_time_s.toFixed(2)}s</span>
						{:else if last_geom_stats}
							<span class="stat">{last_geom_stats.n_entities} ent · {last_geom_stats.n_triangles.toLocaleString()} tris</span>
						{/if}
					</div>
					<div class="editor-wrap">
						<CodeEditor bind:value={code} onSave={on_save} />
					</div>
					<Resizer vertical onStart={on_output_drag_start} onDelta={on_output_resize} />
					<div
						class="output"
						class:collapsed={output_collapsed}
						style:height={output_collapsed ? '24px' : `${output_h}px`}
					>
						<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
						<div
							class="output-head"
							onclick={output_collapsed ? () => { output_collapsed = false; output_h = Math.max(output_h, 140); output_track = output_h; save_layout(); } : undefined}
						>
							<span class="output-title">Output</span>
							{#if log_lines.length}
								<span class="output-count">{log_lines.length}</span>
							{/if}
							<span class="spacer"></span>
							{#if log_lines.length && !output_collapsed}
								<button class="tb" onclick={() => (log_lines = [])} title="Clear" aria-label="Clear">
									<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
										<polyline points="3,5 13,5" />
										<path d="M6 5V3h4v2" />
										<path d="M5 5l1 8h4l1-8" />
									</svg>
								</button>
							{/if}
						</div>
						{#if !output_collapsed}
							<div class="output-body">
								{#each log_lines as line}
									<div class="line">{line}</div>
								{:else}
									<div class="empty">No output yet — Ctrl+S to run, or Generate Mesh / Run Simulation.</div>
								{/each}
							</div>
						{/if}
					</div>
				</div>
			{/if}
		</aside>

		<Resizer onStart={on_editor_drag_start} onDelta={on_editor_resize} />

		<section class="pane viewer-pane" style:flex={viewer_collapsed ? `0 0 ${COLLAPSED_W}px` : '1 1 0'}>
			{#if viewer_collapsed}
				<button
					class="collapsed-strip"
					title="Click or drag to expand"
					aria-label="Expand viewer"
					onclick={() => { viewer_collapsed = false; editor_track = editor_w; save_layout(); }}
				>
					<span class="strip-label">Viewer</span>
				</button>
			{:else}
				<div class="pane-inner">
					<nav class="tabs">
						<button class="tab-btn" class:active={display === 'view3d'} onclick={() => (display = 'view3d')}>3D</button>
						<button class="tab-btn" class:active={display === 'plots'} onclick={() => (display = 'plots')}>S-Params</button>
						{#if display === 'view3d'}
							<span class="tab-spacer"></span>
							<div class="layer-toggles">
								<button class="layer-toggle" class:active={show_geometry} onclick={() => (show_geometry = !show_geometry)} title="Geometry surfaces (G)">Geometry</button>
								<button class="layer-toggle" class:active={show_wireframe} onclick={() => (show_wireframe = !show_wireframe)} title="Mesh wireframe (M)">Mesh</button>
								<button class="layer-toggle" class:active={show_field} disabled={!last_solve_stats} onclick={() => (show_field = !show_field)} title="Field cloud (E)">Field</button>
							</div>
						{/if}
					</nav>
					<div class="viewer-slot">
						{#if display === 'view3d'}
							<MeshViewer
								bind:this={viewer}
								mesh={mesh_data}
								{show_geometry}
								{show_wireframe}
								{show_field}
								field={show_field ? field_abc : null}
							/>
						{:else}
							<ResultsPanel {freqs} {smats} metrics={[]} />
						{/if}
					</div>
					{#if display === 'view3d' && show_field && fields_raw && freqs.length}
						<div class="field-controls">
							<label class="field-ctrl">
								<span class="lbl">Freq</span>
								<input type="range" min="0" max={freqs.length - 1} step="1" bind:value={field_freq_idx} />
								<span class="val">{(freqs[field_freq_idx] / 1e9).toFixed(2)} GHz</span>
							</label>
							{#if fields_raw[field_freq_idx] && fields_raw[field_freq_idx].length > 1}
								<label class="field-ctrl">
									<span class="lbl">Excitation</span>
									<select bind:value={field_port_idx}>
										{#each fields_raw[field_freq_idx] as _f, pi}
											<option value={pi}>Port {pi + 1}</option>
										{/each}
									</select>
								</label>
							{/if}
						</div>
					{/if}
				</div>
			{/if}
		</section>
	</main>

</div>

<style>
	.app {
		display: flex;
		flex-direction: column;
		height: 100vh;
		background: var(--bg);
		color: var(--text);
		font-family: var(--font-body);
	}

	header {
		display: flex;
		align-items: center;
		padding: 0 var(--space-xl);
		gap: var(--space-md);
		height: 36px;
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		flex-shrink: 0;
	}
	header .brand {
		color: var(--accent);
		font-family: var(--font-mono);
		font-weight: 700;
		letter-spacing: 0.5px;
		text-transform: uppercase;
		font-size: var(--fs-xs);
	}
	header .nav-sep {
		width: 1px;
		height: 100%;
		background: var(--border);
		flex-shrink: 0;
	}
	header .path-sep {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	header .workdir,
	header .active-file {
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	header .active-file { color: var(--text); }
	header .status {
		margin-left: auto;
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}

	main {
		display: flex;
		flex-direction: row;
		flex: 1;
		min-height: 0;
		overflow: hidden;
	}

	.pane {
		display: flex;
		flex-direction: column;
		min-width: 0;
		min-height: 0;
		overflow: hidden;
		background: var(--bg);
	}
	.pane-inner {
		display: flex;
		flex-direction: column;
		flex: 1;
		min-height: 0;
		overflow: hidden;
	}

	.collapsed-strip {
		flex: 1;
		display: flex;
		align-items: flex-start;
		justify-content: center;
		padding: var(--space-lg) 0;
		background: var(--bg-surface);
		color: var(--text-muted);
		border: 0;
		width: 100%;
		cursor: pointer;
		text-transform: none;
		letter-spacing: 0;
		font-weight: normal;
		transition: color var(--transition), background var(--transition);
	}
	.collapsed-strip:hover {
		background: var(--bg-panel);
		color: var(--accent);
	}
	.output.collapsed .output-head { cursor: pointer; }
	.output.collapsed .output-head:hover { background: var(--bg-panel); }
	.strip-label {
		writing-mode: vertical-rl;
		transform: rotate(180deg);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		letter-spacing: 1px;
		text-transform: uppercase;
		font-weight: 600;
	}

	.files-pane { background: var(--bg-surface); }
	.editor-pane { background: var(--bg); }
	.viewer-pane { background: var(--canvas-bg); }

	.toolbar {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		padding: 0 var(--space-lg);
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		height: 36px;
		flex-shrink: 0;
	}
	.toolbar .hint {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.toolbar .spacer { flex: 1; }
	.toolbar .stat {
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.toolbar button.primary {
		height: 24px;
		padding: 0 var(--space-lg);
		font-size: var(--fs-xs);
		letter-spacing: 0.5px;
	}
	.toolbar button.primary:disabled {
		background: var(--bg-panel);
		color: var(--text-dim);
		cursor: default;
	}

	.editor-wrap {
		flex: 1;
		min-height: 0;
		overflow: hidden;
	}

	.tabs {
		display: flex;
		gap: 0;
		padding: 0;
		border-bottom: 1px solid var(--border);
		background: var(--bg-surface);
		height: 36px;
		flex-shrink: 0;
		align-items: stretch;
	}
	.tabs .tab-btn {
		background: transparent;
		color: var(--text-dim);
		border: 0;
		padding: 0 var(--space-lg);
		cursor: pointer;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 600;
		letter-spacing: 0.5px;
		text-transform: uppercase;
		transition: color var(--transition);
	}
	.tabs .tab-btn:hover { color: var(--text-muted); }
	.tabs .tab-btn.active { color: var(--accent); }

	.tb.collapse {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 22px;
		height: 22px;
		padding: 0;
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text-muted);
		cursor: pointer;
		text-transform: none;
		letter-spacing: 0;
		font-weight: normal;
		transition: background var(--transition), border-color var(--transition), color var(--transition);
	}
	.tb.collapse:hover { background: var(--bg-panel); border-color: var(--accent); color: var(--text); }

	.tab-spacer { flex: 1; }

	.layer-toggles {
		display: flex;
		align-items: center;
		gap: 0;
		padding-right: var(--space-md);
		border-left: 1px solid var(--border);
		margin-left: 0;
	}
	.layer-toggle {
		background: transparent;
		color: var(--text-dim);
		border: 0;
		border-right: 1px solid var(--border-subtle);
		padding: 0 var(--space-md);
		height: 22px;
		margin: 7px 0 7px var(--space-md);
		cursor: pointer;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 500;
		letter-spacing: 0.5px;
		text-transform: uppercase;
		transition: color var(--transition);
	}
	.layer-toggle:last-child { border-right: 0; }
	.layer-toggle:hover { color: var(--text-muted); }
	.layer-toggle.active { color: var(--accent); }
	.layer-toggle:disabled { color: var(--text-dim); cursor: default; opacity: 0.5; }

	.field-controls {
		display: flex;
		align-items: center;
		gap: var(--space-xl);
		padding: var(--space-sm) var(--space-lg);
		background: var(--bg-surface);
		border-top: 1px solid var(--border-subtle);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		flex-shrink: 0;
	}
	.field-ctrl {
		display: flex;
		align-items: center;
		gap: var(--space-md);
	}
	.field-ctrl .lbl {
		color: var(--text-dim);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		min-width: 36px;
	}
	.field-ctrl .val {
		color: var(--accent);
		min-width: 80px;
	}
	.field-ctrl input[type="range"] {
		width: 200px;
		accent-color: var(--accent);
	}
	.field-ctrl select {
		width: auto;
	}

	.viewer-slot {
		flex: 1;
		min-height: 0;
		display: flex;
		flex-direction: column;
	}
	.viewer-slot > :global(*) { flex: 1; min-height: 0; }

	.output {
		display: flex;
		flex-direction: column;
		background: var(--bg-inset);
		border-top: 1px solid var(--border);
		flex-shrink: 0;
		min-height: 28px;
		overflow: hidden;
	}
	.output-head {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		padding: 0 var(--space-lg);
		height: 28px;
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border-subtle);
		flex-shrink: 0;
	}
	.output-title {
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-weight: 600;
	}
	.output-count {
		background: var(--accent-dim);
		color: var(--accent);
		font-family: var(--font-mono);
		font-size: 10px;
		padding: 1px 6px;
		min-width: 18px;
		text-align: center;
	}
	.output-head .spacer { flex: 1; }
	.output-body {
		flex: 1;
		overflow: auto;
		padding: var(--space-sm) var(--space-lg);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
		background: var(--bg-inset);
	}
	.output-body .line { white-space: pre-wrap; word-break: break-word; padding: 1px 0; }
	.output-body .empty { color: var(--text-dim); font-style: italic; }
</style>
