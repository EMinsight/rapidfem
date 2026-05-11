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

	let status = $state('idle');
	let workdir = $state('');
	let active_path = $state<string | null>(null);
	let code = $state('');
	let dirty = $state(false);
	let log_lines = $state<string[]>([]);

	let mesh_data = $state<MeshData | null>(null);
	let smats = $state<SMatrix[]>([]);
	let freqs = $state<number[]>([]);

	let geom_busy = $state(false);
	let mesh_busy = $state(false);
	let solve_busy = $state(false);
	let last_geom_stats = $state<{ n_entities: number; n_triangles: number } | null>(null);
	let last_mesh_stats = $state<{ n_tets: number; n_tris: number; mesh_time_s: number } | null>(null);
	let last_solve_stats = $state<{ n_freq: number; n_dofs: number; solve_time_s: number } | null>(null);

	let display = $state<'view3d' | 'plots'>('view3d');
	let unsub_bus: (() => void) | null = null;
	let geom_debounce: ReturnType<typeof setTimeout> | null = null;

	onMount(async () => {
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

<div class="app">
	<header>
		<span class="brand">rapidfem</span>
		<span class="sep">/</span>
		<span class="workdir">{workdir}</span>
		{#if active_path}
			<span class="sep">/</span>
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
		</aside>

		<section class="viewer-pane">
			<nav class="tabs">
				<button class:active={display === 'view3d'} onclick={() => (display = 'view3d')}>3D View</button>
				<button class:active={display === 'plots'} onclick={() => (display = 'plots')}>S-Parameters</button>
			</nav>
			<div class="viewer-slot">
				{#if display === 'view3d'}
					<MeshViewer mesh={mesh_data} show_geometry={true} show_wireframe={false} show_field={false} />
				{:else}
					<ResultsPanel {freqs} {smats} metrics={[]} />
				{/if}
			</div>
		</section>
	</main>

	<footer class="log">
		<div class="log-title">Output</div>
		<div class="log-body">
			{#each log_lines as line}
				<div class="line">{line}</div>
			{/each}
		</div>
	</footer>
</div>

<style>
	.app {
		display: grid;
		grid-template-rows: 36px 1fr 160px;
		height: 100vh;
		background: var(--bg);
		color: var(--text);
		font-family: var(--font-body);
	}

	header {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		padding: 0 var(--space-xl);
		border-bottom: 1px solid var(--border);
		background: var(--bg-surface);
		font-size: var(--fs-sm);
	}
	header .brand {
		color: var(--accent);
		font-weight: 700;
		letter-spacing: 0.5px;
		text-transform: uppercase;
		font-size: var(--fs-sm);
	}
	header .sep { color: var(--text-dim); }
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
		display: grid;
		grid-template-columns: 220px 1fr 1fr;
		min-height: 0;
	}

	.files-pane {
		border-right: 1px solid var(--border);
		min-height: 0;
		background: var(--bg-surface);
	}

	.editor-pane {
		display: flex;
		flex-direction: column;
		border-right: 1px solid var(--border);
		min-width: 0;
		min-height: 0;
		background: var(--bg);
	}

	.toolbar {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		padding: var(--space-md) var(--space-lg);
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		min-height: 38px;
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
		padding: 4px 10px;
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

	.viewer-pane {
		display: flex;
		flex-direction: column;
		min-height: 0;
		min-width: 0;
		background: var(--canvas-bg);
	}

	.tabs {
		display: flex;
		gap: 0;
		padding: 0;
		border-bottom: 1px solid var(--border);
		background: var(--bg-surface);
		min-height: 38px;
		align-items: stretch;
	}
	.tabs button {
		background: transparent;
		color: var(--text-muted);
		border: 0;
		border-bottom: 2px solid transparent;
		padding: 0 var(--space-xl);
		cursor: pointer;
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-weight: 600;
		font-family: var(--font-body);
	}
	.tabs button:hover { color: var(--text); background: transparent; }
	.tabs button.active {
		color: var(--accent);
		border-bottom-color: var(--accent);
		background: transparent;
	}

	.viewer-slot {
		flex: 1;
		min-height: 0;
		display: flex;
		flex-direction: column;
	}
	.viewer-slot > :global(*) { flex: 1; min-height: 0; }

	footer.log {
		display: flex;
		flex-direction: column;
		background: var(--bg-inset);
		border-top: 1px solid var(--border);
		min-height: 0;
	}
	.log-title {
		padding: var(--space-sm) var(--space-lg);
		color: var(--text-muted);
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		border-bottom: 1px solid var(--border-subtle);
	}
	.log-body {
		flex: 1;
		overflow: auto;
		padding: var(--space-sm) var(--space-lg);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
	}
	.log-body .line { white-space: pre; }
</style>
