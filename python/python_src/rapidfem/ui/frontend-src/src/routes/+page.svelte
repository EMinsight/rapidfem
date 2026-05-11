<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { runCode, meshGeometry, solve, listFiles, readFile, writeFile, subscribeBus, meshPayloadToMeshData, sparamsToSMatrices, health, type RunResponse, type MeshResponse, type SolveResponse, type SMatrix, type BusEvent } from '$lib/api';
	import type { MeshData } from '$lib/msh';
	import MeshViewer from '$lib/components/MeshViewer.svelte';
	import ResultsPanel from '$lib/components/ResultsPanel.svelte';
	import CodeEditor from '$lib/components/CodeEditor.svelte';
	import FileBrowser from '$lib/components/FileBrowser.svelte';

	let status = $state('idle');
	let workdir = $state('');
	let active_path = $state<string | null>(null);
	let code = $state(
		'import rapidfem\n\n' +
		'g = rapidfem.Geometry()\n' +
		'g.box(60e-3, 60e-3, 1.6e-3)\n' +
		'rapidfem.show(g)\n',
	);
	let log_lines = $state<string[]>([]);

	let mesh_data = $state<MeshData | null>(null);
	let smats = $state<SMatrix[]>([]);
	let freqs = $state<number[]>([]);

	// Independent busy flags per stage so the user can see which one is live.
	let geom_busy = $state(false);
	let mesh_busy = $state(false);
	let solve_busy = $state(false);
	let last_geom_stats = $state<{ n_entities: number; n_triangles: number } | null>(null);
	let last_mesh_stats = $state<{ n_tets: number; n_tris: number; mesh_time_s: number } | null>(null);
	let last_solve_stats = $state<{ n_freq: number; n_dofs: number; solve_time_s: number } | null>(null);

	let display = $state<'view3d' | 'plots'>('view3d');
	let unsub_bus: (() => void) | null = null;
	let geom_debounce: ReturnType<typeof setTimeout> | null = null;
	let dirty = $state(false);  // editor diverged from disk

	onMount(async () => {
		try {
			const h = await health();
			workdir = h.workdir;
		} catch (e) {
			status = 'backend unreachable';
		}
		unsub_bus = subscribeBus((e: BusEvent) => {
			if (e.kind === 'stage_start') status = `${e.stage}…`;
			else if (e.kind === 'stage_end') status = e.ok ? `${e.stage} ok` : `${e.stage} failed`;
		});
		// Restore last-opened file if any.
		const last = localStorage.getItem('rapidfem.active_path');
		if (last) await open_file(last);
	});

	onDestroy(() => unsub_bus?.());

	function append_log(label: string, r: RunResponse | MeshResponse | SolveResponse) {
		if (r.stdout) log_lines = [...log_lines, ...r.stdout.split('\n').filter(Boolean).map((l) => `[${label}] ${l}`)];
		if (r.stderr) log_lines = [...log_lines, ...r.stderr.split('\n').filter(Boolean).map((l) => `[${label} err] ${l}`)];
		if (!r.ok && r.error) log_lines = [...log_lines, `[${label}] ${r.error.type}: ${r.error.message}`];
	}

	async function open_file(path: string) {
		try {
			const content = await readFile(path);
			code = content;
			active_path = path;
			dirty = false;
			localStorage.setItem('rapidfem.active_path', path);
			// Refresh the geometry view to match the file we just opened.
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

	async function on_save(text: string) {
		code = text;
		dirty = false;
		// Persist to disk when there is an active file.
		if (active_path) {
			try {
				await writeFile(active_path, text);
			} catch (e) {
				log_lines = [...log_lines, `[save] ${e}`];
			}
		}
		// Auto-update geometry view, debounced so a fast Ctrl+S burst coalesces.
		if (geom_debounce) clearTimeout(geom_debounce);
		geom_debounce = setTimeout(() => {
			geom_debounce = null;
			void run_geometry();
		}, 200);
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
					// Project the geometry payload into the viewer's MeshData
					// shape: every entity contributes its own surface tris.
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
		display = 'plots';
		try {
			const r = await solve(code);
			append_log('solve', r);
			if (r.ok && r.result) {
				freqs = r.result.frequencies;
				smats = sparamsToSMatrices(r.result.sparams);
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

	// Track unsaved edits.
	$effect(() => {
		if (code) dirty = true;
	});

	// Build a MeshData-equivalent from a geometry payload (per-entity surface
	// tris). The MeshViewer is happy with tris+phys-names — tets stay empty.
	function geometryToMeshData(p: import('$lib/api').GeometryPayload): MeshData {
		// Concatenate positions across entities; assign each a synthetic phys
		// tag so the viewer's per-tag color toggle still works.
		let total_tris = 0;
		const phys_names = new Map<number, string>();
		const phys_dim = new Map<number, number>();
		for (let i = 0; i < p.entities.length; i++) total_tris += p.entities[i].positions.length / 9;

		const nodes_flat: number[] = [];
		const tris_flat: number[] = [];
		const tri_phys_flat: number[] = [];
		let next_node = 0;
		p.entities.forEach((ent, i) => {
			const tag = i + 1;
			phys_names.set(tag, ent.name);
			phys_dim.set(tag, ent.dim);
			// positions stored flat: 9 floats per triangle.
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
			bbox: { min: [...p.bbox.min] as [number, number, number], max: [...p.bbox.max] as [number, number, number] },
		};
	}
</script>

<svelte:head>
	<title>rapidfem</title>
</svelte:head>

<div class="app">
	<header>
		<span class="brand">rapidfem</span>
		<span class="workdir">{workdir}</span>
		{#if active_path}
			<span class="active-file">{active_path}{dirty ? ' •' : ''}</span>
		{/if}
		<span class="status">{status}</span>
	</header>

	<main>
		<aside class="files-pane">
			<FileBrowser bind:active_path={active_path} onOpen={open_file} onNew={new_file} />
		</aside>
		<aside class="editor-pane">
			<div class="toolbar">
				<span class="hint">Ctrl+S → geometry preview</span>
				<button onclick={on_mesh} disabled={mesh_busy} title="Run gmsh + show full tet mesh">
					{mesh_busy ? 'meshing…' : 'Generate Mesh'}
				</button>
				<button onclick={on_solve} disabled={solve_busy} title="Run FEM frequency sweep">
					{solve_busy ? 'solving…' : 'Run Simulation'}
				</button>
				<span class="spacer"></span>
				{#if last_mesh_stats}
					<span class="stat">mesh: {last_mesh_stats.n_tets.toLocaleString()} tets · {last_mesh_stats.mesh_time_s.toFixed(2)}s</span>
				{:else if last_geom_stats}
					<span class="stat">geom: {last_geom_stats.n_entities} ent · {last_geom_stats.n_triangles.toLocaleString()} tris</span>
				{/if}
				{#if last_solve_stats}
					<span class="stat">solve: {last_solve_stats.n_freq} freq · {last_solve_stats.solve_time_s.toFixed(2)}s</span>
				{/if}
			</div>
			<div class="editor-wrap">
				<CodeEditor bind:value={code} onSave={on_save} />
			</div>
		</aside>

		<section class="viewer-pane">
			<nav class="tabs">
				<button class:active={display === 'view3d'} onclick={() => (display = 'view3d')}>3D</button>
				<button class:active={display === 'plots'} onclick={() => (display = 'plots')}>S-params</button>
			</nav>
			{#if display === 'view3d'}
				<MeshViewer mesh={mesh_data} show_geometry={true} show_wireframe={false} show_field={false} />
			{:else}
				<ResultsPanel {freqs} {smats} metrics={[]} />
			{/if}
		</section>
	</main>

	<footer class="log">
		{#each log_lines as line}
			<div class="line">{line}</div>
		{/each}
	</footer>
</div>

<style>
	.app { display: grid; grid-template-rows: 36px 1fr 140px; height: 100vh; font: 13px/1.4 system-ui, sans-serif; }
	header { display: flex; align-items: center; gap: 16px; padding: 0 12px; border-bottom: 1px solid #2a2a2a; background: #1a1a1a; color: #ddd; }
	.brand { font-weight: 600; }
	.workdir { color: #888; font-family: monospace; }
	.active-file { color: #ccc; font-family: monospace; }
	.status { margin-left: auto; color: #aaa; }
	.hint { color: #666; font-size: 11px; align-self: center; margin-right: 6px; }
	.stat { color: #88c; font-size: 11px; align-self: center; }
	.spacer { flex: 1; }
	main { display: grid; grid-template-columns: 200px 1fr 1fr; min-height: 0; }
	.files-pane { border-right: 1px solid #2a2a2a; min-height: 0; }
	.editor-pane { display: flex; flex-direction: column; border-right: 1px solid #2a2a2a; min-width: 0; }
	.toolbar { display: flex; gap: 8px; padding: 6px 8px; background: #161616; border-bottom: 1px solid #2a2a2a; align-items: center; }
	.toolbar button { background: #2a2a2a; color: #ddd; border: 1px solid #3a3a3a; padding: 4px 10px; cursor: pointer; }
	.toolbar button:disabled { opacity: 0.5; cursor: default; }
	.editor-wrap { flex: 1; min-height: 0; }
	.viewer-pane { display: flex; flex-direction: column; min-height: 0; }
	.tabs { display: flex; gap: 4px; padding: 4px 8px; border-bottom: 1px solid #2a2a2a; background: #161616; }
	.tabs button { background: transparent; color: #aaa; border: 0; padding: 4px 10px; cursor: pointer; }
	.tabs button.active { color: #e8e8e8; border-bottom: 2px solid #5a8; }
	footer.log { background: #0a0a0a; color: #aaa; border-top: 1px solid #2a2a2a; padding: 6px 10px; overflow: auto; font: 12px ui-monospace, Consolas, monospace; }
	footer.log .line { white-space: pre; }
</style>
