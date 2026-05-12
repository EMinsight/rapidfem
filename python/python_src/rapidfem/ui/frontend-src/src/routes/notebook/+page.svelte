<script lang="ts">
	import { onMount } from 'svelte';
	import {
		readFile, writeFile, readExample,
		meshPayloadToMeshData, sparamsToSMatrices, health,
		type MeshPayload, type GeometryPayload, type SMatrix,
	} from '$lib/api';
	import { get_kernel, type SolveResultPayload } from '$lib/kernel';
	import { IS_STATIC_MODE } from '$lib/static_mode';
	import type { MeshData } from '$lib/msh';
	import MeshViewer from '$lib/components/MeshViewer.svelte';
	import ResultsPanel from '$lib/components/ResultsPanel.svelte';
	import Notebook from '$lib/components/Notebook.svelte';
	import FileBrowser from '$lib/components/FileBrowser.svelte';
	import Resizer from '$lib/components/Resizer.svelte';
	import Select from '$lib/components/Select.svelte';

	let status = $state('idle');
	let workdir = $state('');
	let active_path = $state<string | null>(null);
	let code = $state('');
	let dirty = $state(false);
	let log_lines = $state<string[]>([]);
	let output_body_el: HTMLElement | undefined = $state();
	$effect(() => {
		// Auto-scroll to bottom when new lines arrive (unless user has
		// scrolled up: bail when they're more than ~80px from the bottom).
		const n = log_lines.length;
		if (!output_body_el || n === 0) return;
		const el = output_body_el;
		const at_bottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
		if (at_bottom) queueMicrotask(() => { el.scrollTop = el.scrollHeight; });
	});

	let mesh_data = $state<MeshData | null>(null);
	let wireframe = $state<import('$lib/api').GeometryPayload | null>(null);
	let smats = $state<SMatrix[]>([]);
	let freqs = $state<number[]>([]);
	let fields_raw = $state<(number[] | null)[][] | null>(null);
	let field_freq_idx = $state(0);
	let field_port_idx = $state(0);
	let field_density = $state(3);
	let field_scale_mode = $state<'log' | 'lin'>('lin');
	let field_abc = $derived<Float32Array | null>(
		fields_raw && fields_raw[field_freq_idx] && fields_raw[field_freq_idx][field_port_idx]
			? new Float32Array(fields_raw[field_freq_idx][field_port_idx] as number[])
			: null,
	);

	let last_geom_stats = $state<{ n_entities: number; n_triangles: number } | null>(null);
	let last_mesh_stats = $state<{ n_tets: number; n_tris: number; mesh_time_s: number } | null>(null);
	let last_solve_stats = $state<{ n_freq: number; n_dofs: number; solve_time_s: number } | null>(null);

	let display = $state<'view3d' | 'plots'>('view3d');
	let show_geometry = $state(true);
	let show_wireframe = $state(false);
	let show_field = $state(false);
	let viewer: ReturnType<typeof MeshViewer> | undefined = $state();

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
		if (IS_STATIC_MODE) {
			// No backend — populate workdir label, then resolve which example
			// to open. Priority: ?example= URL query > last localStorage entry
			// > first manifest entry. The query path lets the landing page
			// link straight into a specific example.
			workdir = '(static demo)';
			const params = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
			const wanted = params?.get('example') ?? null;
			const last = localStorage.getItem('rapidfem.active_path');
			try {
				if (wanted) {
					const filename = wanted.endsWith('.py') ? wanted : `${wanted}.py`;
					await open_example(filename);
				} else if (last) {
					await open_example(last);
				} else {
					// Pick the first example from the manifest.
					const r = await fetch(`${(import.meta.env.BASE_URL ?? '/')}demo/manifest.json`.replace(/\/+/g, '/'));
					if (r.ok) {
						const m = await r.json();
						if (m.examples?.length) await open_example(m.examples[0].filename);
					}
				}
			} catch (e) {
				log_lines = [...log_lines, `[static-demo] ${e}`];
			}
			return;
		}
		try {
			const h = await health();
			workdir = h.workdir;
		} catch {
			status = 'backend unreachable';
		}
		// Logs + display events flow through KernelClient now (single channel).
		const last = localStorage.getItem('rapidfem.active_path');
		if (last) await open_file(last);
	});


	let notebook: ReturnType<typeof Notebook> | undefined = $state();
	let dirty_save_t: ReturnType<typeof setTimeout> | null = null;

	async function open_file(path: string) {
		try {
			const content = await readFile(path);
			const prev_path = active_path;
			code = content;
			active_path = path;
			dirty = false;
			localStorage.setItem('rapidfem.active_path', path);
			clear_stale_results();
			mesh_data = null;
			log_lines = [];
			if (prev_path !== path) {
				try { get_kernel().reset(path); } catch {}
			}
			// In the static demo the user can't hit Run, so auto-replay the
			// baked cell stream so the viewer is populated on open.
			if (IS_STATIC_MODE) {
				queueMicrotask(() => { void notebook?.run_all_cells(); });
			}
		} catch (e) {
			// 404 typically means a stale localStorage pointer to a file that
			// has been deleted (e.g. the old welcome.py). Clear it silently
			// rather than nag the user with a log line every reload.
			const msg = String(e);
			if (msg.includes('HTTP 404')) {
				localStorage.removeItem('rapidfem.active_path');
				return;
			}
			log_lines = [...log_lines, `[open ${path}] ${e}`];
		}
	}

	async function open_example(name: string) {
		try {
			const content = await readExample(name);
			// Static mode: open the example *in place* without copying it
			// into a workdir (there is no workdir, and no writeFile).
			if (IS_STATIC_MODE) {
				const prev_path = active_path;
				code = content;
				active_path = name;
				dirty = false;
				localStorage.setItem('rapidfem.active_path', name);
				clear_stale_results();
				mesh_data = null;
				log_lines = [];
				if (prev_path !== name) {
					try { get_kernel().reset(name); } catch {}
				}
				queueMicrotask(() => { void notebook?.run_all_cells(); });
				return;
			}
			// Live mode: if the example file is already in the workdir
			// (the default — `rapidfem serve` auto-populates them on first
			// run), just open it. Never overwrite user edits, never spawn
			// `_1.py` clones.
			try {
				await readFile(name);
				await open_file(name);
				return;
			} catch (e) {
				if (!String(e).includes('HTTP 404')) throw e;
			}
			// File missing → write it once.
			await writeFile(name, content);
			await open_file(name);
		} catch (e) {
			log_lines = [...log_lines, `[example ${name}] ${e}`];
		}
	}

	function on_file_closed(path: string) {
		if (active_path === path) {
			active_path = null;
			code = '';
			localStorage.removeItem('rapidfem.active_path');
			clear_stale_results();
			mesh_data = null;
		}
	}

	async function new_file() {
		const name = prompt('New file name (e.g. patch.py):');
		if (!name) return;
		const path = name.endsWith('.py') ? name : `${name}.py`;
		try {
			await writeFile(path, '# %% New rapidfem notebook\nimport rapidfem\n\n');
			await open_file(path);
		} catch (e) {
			log_lines = [...log_lines, `[new ${path}] ${e}`];
		}
	}

	function clear_stale_results() {
		fields_raw = null;
		smats = [];
		freqs = [];
		last_solve_stats = null;
		last_mesh_stats = null;
		last_geom_stats = null;
		show_field = false;
	}

	// Notebook → backend cell exec. Uses the single WS kernel channel —
	// streams logs + display events in order, returns final ok/error.
	async function on_run_cell(cell_source: string, reset_first: boolean): Promise<'ok' | 'error'> {
		if (reset_first) clear_stale_results();
		const cell_id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
		log_lines = [...log_lines, `─── CELL ───`];
		const r = await get_kernel().execute({
			cell_id,
			file: active_path ?? '<unnamed>',
			code: cell_source,
			reset: reset_first,
			onStream: (stream, line) => {
				const prefix = stream === 'stderr' ? '!' : ' ';
				const next = [...log_lines, `${prefix} ${line}`];
				log_lines = next.length > 2000 ? next.slice(-2000) : next;
			},
			onDisplay: (kind, payload) => {
				if (kind === 'geometry') {
					const p = payload as GeometryPayload;
					last_geom_stats = {
						n_entities: p.stats.n_entities,
						n_triangles: p.stats.n_triangles ?? 0,
					};
					if (p.wireframe) {
						wireframe = p;
						mesh_data = null;
					} else {
						wireframe = null;
						mesh_data = geometryToMeshData(p);
					}
				} else if (kind === 'mesh') {
					const m = payload as MeshPayload;
					wireframe = null;
					mesh_data = meshPayloadToMeshData(m);
					last_mesh_stats = {
						n_tets: m.stats.n_tets,
						n_tris: m.stats.n_tris,
						mesh_time_s: m.stats.mesh_time_s,
					};
				} else if (kind === 'result') {
					const res = payload as SolveResultPayload;
					freqs = res.frequencies;
					smats = sparamsToSMatrices(res.sparams);
					fields_raw = res.fields ?? null;
					field_freq_idx = 0;
					field_port_idx = 0;
					last_solve_stats = {
						n_freq: res.n_freq,
						n_dofs: res.n_dofs,
						solve_time_s: res.solve_time_s,
					};
				}
			},
		});
		log_lines = [...log_lines, `↳ cell ${r.ok ? 'ok' : 'failed'}`];
		return r.ok ? 'ok' : 'error';
	}

	function on_reset_kernel() {
		clear_stale_results();
		mesh_data = null;
		get_kernel().reset(active_path ?? '<unnamed>');
		log_lines = [...log_lines, `[kernel] reset`];
	}

	// Debounced save on edits. No auto-exec — Shift+Enter / Run All does that.
	$effect(() => {
		const _ = code;  // track changes
		if (!active_path) return;
		if (IS_STATIC_MODE) return;  // no writes in static-demo mode
		dirty = true;
		if (dirty_save_t) clearTimeout(dirty_save_t);
		dirty_save_t = setTimeout(async () => {
			dirty_save_t = null;
			if (!active_path) return;
			try {
				await writeFile(active_path, code);
				dirty = false;
			} catch (e) {
				log_lines = [...log_lines, `[save] ${e}`];
			}
		}, 600);
	});

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
		<a class="brand" href="https://fem.rapidpassives.org" target="_blank" rel="noopener" title="RapidFEM landing"><img src="/favicon.svg" alt="RapidFEM" class="logo" /></a>
		<span class="nav-sep"></span>
		{#if active_path}
			<span class="active-file has-tip">
				{active_path}{dirty ? ' •' : ''}
				<span class="tip right">{workdir}</span>
			</span>
		{:else}
			<span class="workdir-only">{workdir}</span>
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
						onOpenExample={open_example}
						onClosed={on_file_closed}
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
						<button class="primary has-tip" onclick={() => notebook?.run_all_cells()} disabled={IS_STATIC_MODE}>
							Run All
							<span class="tip">{IS_STATIC_MODE ? 'Disabled in static demo' : 'Run all cells'}<kbd>Ctrl+Shift+Enter</kbd></span>
						</button>
						<button class="primary has-tip" onclick={() => notebook?.run_current_cell()} disabled={IS_STATIC_MODE}>
							Run Cell
							<span class="tip">{IS_STATIC_MODE ? 'Disabled in static demo' : 'Run current cell'}<kbd>Shift+Enter</kbd></span>
						</button>
						<button class="primary subtle has-tip" onclick={on_reset_kernel} disabled={IS_STATIC_MODE}>
							Restart Kernel
							<span class="tip">{IS_STATIC_MODE ? 'Disabled in static demo' : 'Wipe namespace + gmsh state'}</span>
						</button>
					</div>
					<div class="editor-wrap">
						<Notebook
							bind:this={notebook}
							bind:source={code}
							file_path={active_path ?? '<unnamed>'}
							readonly={IS_STATIC_MODE}
							onRunCell={on_run_cell}
						/>
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
							<div class="output-body" bind:this={output_body_el}>
								{#each log_lines as line}
									<div class="line">{line}</div>
								{:else}
									<div class="empty">No output yet — Shift+Enter to run a cell, or hit Run All.</div>
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
						<span class="nav-sep"></span>
						<button class="tab-btn" class:active={display === 'plots'} onclick={() => (display = 'plots')}>S-Params</button>
						{#if display === 'view3d'}
							<span class="tab-spacer"></span>
							<span class="nav-sep"></span>
							<button class="tab-btn small has-tip" class:active={show_geometry} onclick={() => (show_geometry = !show_geometry)}>
								Geometry
								<span class="tip left">Toggle surfaces<kbd>G</kbd></span>
							</button>
							<span class="nav-sep"></span>
							<button class="tab-btn small has-tip" class:active={show_wireframe} disabled={!mesh_data || mesh_data.tets.length === 0} onclick={() => (show_wireframe = !show_wireframe)}>
								Mesh
								<span class="tip left">Toggle tet wireframe<kbd>M</kbd></span>
							</button>
							<span class="nav-sep"></span>
							<button class="tab-btn small has-tip" class:active={show_field} disabled={!last_solve_stats} onclick={() => (show_field = !show_field)}>
								Field
								<span class="tip left">Toggle field cloud<kbd>E</kbd></span>
							</button>
						{/if}
					</nav>
					<div class="viewer-slot">
						{#if display === 'view3d'}
							<MeshViewer
								bind:this={viewer}
								mesh={mesh_data}
								wireframe={wireframe}
								{show_geometry}
								{show_wireframe}
								{show_field}
								field={show_field ? field_abc : null}
								point_density={field_density}
								scale_mode={field_scale_mode}
							/>
						{:else}
							<ResultsPanel {freqs} {smats} metrics={[]} />
						{/if}
					</div>
					{#if display === 'view3d' && show_field && fields_raw && freqs.length}
						<div class="field-controls">
							<label class="field-ctrl">
								<span class="lbl">Freq</span>
								<input class="slider" type="range" min="0" max={freqs.length - 1} step="1" bind:value={field_freq_idx} />
								<span class="val">{(freqs[field_freq_idx] / 1e9).toFixed(2)} GHz</span>
							</label>
							<label class="field-ctrl">
								<span class="lbl">Density</span>
								<input class="slider" type="range" min="1" max="10" step="1" bind:value={field_density} />
								<span class="val">{(field_density * 50).toLocaleString()}k pts</span>
							</label>
							<div class="field-ctrl">
								<span class="lbl">Scale</span>
								<div class="seg">
									<button class:active={field_scale_mode === 'log'} onclick={() => (field_scale_mode = 'log')}>Log</button>
									<button class:active={field_scale_mode === 'lin'} onclick={() => (field_scale_mode = 'lin')}>Lin</button>
								</div>
							</div>
							{#if fields_raw[field_freq_idx] && fields_raw[field_freq_idx].length > 1}
								<div class="field-ctrl">
									<span class="lbl">Excitation</span>
									<Select
										bind:value={field_port_idx}
										open_up
										options={fields_raw[field_freq_idx].map((_f, pi) => ({ value: pi, label: `Port ${pi + 1}` }))}
									/>
								</div>
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
		display: inline-flex;
		align-items: center;
		text-decoration: none;
	}
	header .brand .logo {
		height: 22px;
		width: auto;
		display: block;
	}
	header .nav-sep {
		width: 1px;
		height: 100%;
		background: var(--border);
		flex-shrink: 0;
	}
	header .active-file {
		color: var(--text);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		cursor: default;
	}
	header .workdir-only {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-style: italic;
	}
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
		text-transform: uppercase;
		font-weight: 600;
	}
	.toolbar button.primary.subtle {
		background: transparent;
		color: var(--text-muted);
		border: 1px solid var(--border);
	}
	.toolbar button.primary.subtle:hover {
		background: var(--bg-panel);
		color: var(--text);
		border-color: var(--accent);
	}
	.toolbar .sep {
		width: 1px;
		height: 16px;
		background: var(--border);
		align-self: center;
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
	.tabs .tab-btn:hover { color: var(--text); }
	.tabs .tab-btn.active { color: var(--accent); }
	.tabs .tab-btn.small { padding: 0 var(--space-md); }
	.tabs .tab-btn:disabled { color: var(--text-dim); cursor: default; opacity: 0.5; }
	.tabs .nav-sep {
		width: 1px;
		height: 100%;
		background: var(--border);
		flex-shrink: 0;
	}

	.tb {
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
	.tb:hover { background: var(--bg-panel); border-color: var(--accent); color: var(--text); }
	.tb:disabled {
		opacity: 0.4;
		cursor: default;
		background: transparent;
		border-color: var(--border);
		color: var(--text-dim);
	}
	.tb svg { display: block; }

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
	.field-ctrl select {
		width: auto;
		height: 22px;
		padding: 0 var(--space-md);
	}

	.seg {
		display: inline-flex;
		border: 1px solid var(--input-border);
		height: 22px;
	}
	.seg button {
		background: var(--input-bg);
		color: var(--text-muted);
		border: 0;
		border-right: 1px solid var(--input-border);
		padding: 0 var(--space-md);
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-weight: 600;
		cursor: pointer;
		transition: color var(--transition), background var(--transition);
	}
	.seg button:last-child { border-right: 0; }
	.seg button:hover { color: var(--text); }
	.seg button.active {
		color: var(--accent);
		background: var(--accent-dim);
	}

	/* Range slider — flat, themed, no native rounded thumb. */
	.slider {
		-webkit-appearance: none;
		appearance: none;
		width: 160px;
		height: 18px;
		background: transparent;
		cursor: pointer;
		padding: 0;
		margin: 0;
	}
	.slider:focus { outline: none; }
	.slider::-webkit-slider-runnable-track {
		height: 2px;
		background: var(--border);
		border: 0;
	}
	.slider::-moz-range-track {
		height: 2px;
		background: var(--border);
		border: 0;
	}
	.slider::-webkit-slider-thumb {
		-webkit-appearance: none;
		appearance: none;
		width: 10px;
		height: 14px;
		margin-top: -6px;
		background: var(--accent);
		border: 0;
		border-radius: 0;
		cursor: grab;
	}
	.slider::-moz-range-thumb {
		width: 10px;
		height: 14px;
		background: var(--accent);
		border: 0;
		border-radius: 0;
		cursor: grab;
	}
	.slider:hover::-webkit-slider-thumb { background: var(--accent-hover); }
	.slider:hover::-moz-range-thumb { background: var(--accent-hover); }
	.slider:active::-webkit-slider-thumb { cursor: grabbing; }
	.slider:active::-moz-range-thumb { cursor: grabbing; }

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
		background: var(--bg-mid);
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
		background: var(--bg-panel);
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		padding: 1px 6px;
		min-width: 18px;
		text-align: center;
		border: 1px solid var(--border-subtle);
	}
	.output-head .spacer { flex: 1; }
	.output-body {
		flex: 1;
		overflow: auto;
		padding: var(--space-sm) var(--space-lg);
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		color: var(--text-muted);
		background: var(--bg-inset);
	}
	.output-body .line { white-space: pre-wrap; word-break: break-word; padding: 1px 0; }
	.output-body .empty { color: var(--text-dim); font-style: italic; }
</style>
