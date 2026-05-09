<script lang="ts">
	import '$lib/components/fields.css';
	import { EXAMPLES, type DemoExample } from '$lib/examples';
	import { run_streaming_sweep, abort_solver, type SMatrix } from '$lib/wasm';
	import ParamSidebar from '$lib/components/ParamSidebar.svelte';
	import ResultsPanel from '$lib/components/ResultsPanel.svelte';
	import StatusPanel from '$lib/components/StatusPanel.svelte';
	import ExampleSelect from '$lib/components/ExampleSelect.svelte';
	import MeshViewer from '$lib/components/MeshViewer.svelte';
	import { parse_msh, type MeshData } from '$lib/msh';
	import { L_eq_pH, Q_factor, find_srf } from '$lib/sparams';

	let selected_id = $state('spiral');
	let example = $derived<DemoExample>(EXAMPLES[selected_id]);

	let running = $state(false);
	let status = $state('idle');
	let progress = $state(0);
	let log_lines = $state<string[]>([]);

	let freqs = $state<number[]>([]);
	let smats = $state<SMatrix[]>([]);

	let mesh_data = $state<MeshData | null>(null);
	type Display = 'geometry' | 'mesh' | 'both' | 'field' | 'plots';
	let display = $state<Display>('geometry');
	let viewer: MeshViewer | undefined = $state();
	// Field viz: which freq + excitation to show
	let field_freq_idx = $state(0);
	let field_exc_idx = $state(0);
	let field_results = $state<{ freq_hz: number; fields: Float32Array[] }[]>([]);
	let active_field = $derived<Float32Array | null>(
		field_results[field_freq_idx]?.fields[field_exc_idx] ?? null
	);

	// Resizable sidebar
	let sidebar_width = $state(280);
	let dragging_sidebar = false;
	function on_sidebar_drag_start(e: PointerEvent) {
		dragging_sidebar = true;
		(e.target as HTMLElement).setPointerCapture(e.pointerId);
	}
	function on_sidebar_drag(e: PointerEvent) {
		if (!dragging_sidebar) return;
		sidebar_width = Math.max(240, Math.min(500, e.clientX));
	}
	function on_sidebar_drag_end() {
		dragging_sidebar = false;
	}

	function on_keydown(e: KeyboardEvent) {
		if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
		if (display === 'plots') return; // Viewer shortcuts only when viewer active
		switch (e.key) {
			case 'f': case 'F': viewer?.fit_view(); break;
			case '+': case '=': viewer?.zoom_in(); break;
			case '-': case '_': viewer?.zoom_out(); break;
			case 'r': case 'R': viewer?.rotate_90(); break;
			case 'z': case 'Z': viewer?.flip_z(); break;
			case 's': case 'S':
				if (e.ctrlKey || e.metaKey) {
					e.preventDefault();
					viewer?.save_png();
				}
				break;
		}
	}

	let abort_controller: AbortController | null = null;

	// Auto-load mesh when example changes so the geometry is visible before run
	$effect(() => {
		const url = example.msh_url;
		(async () => {
			try {
				const t = await fetch(url).then((r) => r.text());
				mesh_data = parse_msh(t);
			} catch (e) {
				console.error('mesh parse failed', e);
				mesh_data = null;
			}
		})();
	});

	function log(msg: string) {
		log_lines = [...log_lines, msg];
	}

	async function run() {
		if (running) return;
		running = true;
		status = 'loading mesh…';
		progress = 0;
		log_lines = [];
		smats = [];
		freqs = [];
		field_results = [];
		display = 'plots';
		abort_controller = new AbortController();

		try {
			const ex = example;
			const [mesh_resp, toml_resp] = await Promise.all([
				fetch(ex.msh_url),
				fetch(ex.toml_url)
			]);
			const mesh_bytes = new Uint8Array(await mesh_resp.arrayBuffer());
			const config_toml = await toml_resp.text();
			log(`${ex.label} · ${ex.frequencies_hz.length} freqs`);

			const t_start = performance.now();
			await run_streaming_sweep({
				mesh_bytes,
				config_toml,
				frequencies_hz: ex.frequencies_hz,
				abort_signal: abort_controller.signal,
				on_status: (m) => (status = m),
				on_point: (k, total, point) => {
					freqs = [...freqs, point.freq_hz];
					smats = [...smats, point.S];
					field_results = [...field_results, { freq_hz: point.freq_hz, fields: point.fields }];
					progress = (k + 1) / total;
				}
			});
			const dt_total = (performance.now() - t_start) / 1000;
			log(`done in ${dt_total.toFixed(1)}s`);

			if (smats.length > 0) {
				const m = example.metrics;
				if (m.includes('L_eq')) {
					const L0 = L_eq_pH(smats[0], freqs[0]);
					log(`L_eq(${(freqs[0] / 1e9).toFixed(1)} GHz) = ${L0.toFixed(1)} pH`);
					const fSRF = find_srf(freqs, smats);
					if (fSRF != null) log(`SRF ≈ ${(fSRF / 1e9).toFixed(1)} GHz`);
				}
				if (m.includes('Q')) {
					const Q0 = Q_factor(smats[0]);
					log(`Q(${(freqs[0] / 1e9).toFixed(1)} GHz) = ${Q0.toFixed(1)}`);
				}
			}
			status = 'done';
		} catch (e) {
			status = 'failed';
			log(`error: ${e}`);
			console.error(e);
		} finally {
			running = false;
			abort_controller = null;
		}
	}

	function abort() {
		abort_controller?.abort();
		abort_solver();   // terminates the worker so the in-flight LU stops immediately
		status = 'aborted';
		running = false;
	}
</script>

<svelte:head>
	<title>rapidfem — in-browser FEM</title>
	<meta name="description" content="WebAssembly-powered FEM EM solver. Solve Sky130 microstrip and spiral inductor S-parameters and L_eq(f) in your browser, with no backend." />
</svelte:head>

<svelte:window onkeydown={on_keydown} />

<div class="app">
	<header>
		<a class="brand" href="/">
			<span class="brand-text">rapidfem</span>
		</a>
		<span class="nav-sep"></span>
		<nav class="tabs">
			<a class="tab active" href="/">Demo</a>
			<a class="tab" href="https://github.com/milanofthe/rapidfem" target="_blank" rel="noopener">GitHub</a>
		</nav>
	</header>

	<div class="body">
		<aside class="sidebar" style="width: {sidebar_width}px; min-width: {sidebar_width}px;">
			<ParamSidebar>
				<div class="param-section">
					<h4>Example</h4>
					<ExampleSelect bind:value={selected_id} />
					<div class="desc">{example.description}</div>
				</div>

				<div class="param-section">
					<h4>Frequencies</h4>
					<div class="freq-list">
						{#each example.frequencies_hz as f}
							<span class="freq-chip">{(f / 1e9).toFixed(f >= 1e9 ? 1 : 2)} GHz</span>
						{/each}
					</div>
				</div>

				<StatusPanel {running} {status} {progress} {log_lines} onrun={run} onabort={abort} />
			</ParamSidebar>
		</aside>

		<div
			class="resize-handle-v"
			onpointerdown={on_sidebar_drag_start}
			onpointermove={on_sidebar_drag}
			onpointerup={on_sidebar_drag_end}
			role="separator"
			aria-label="Resize sidebar"
			tabindex="-1"
		></div>
		<main class="results-area">
			<div class="view-tabs">
				<button class="vt" class:active={display === 'geometry'} onclick={() => (display = 'geometry')}>Geometry</button>
				<button class="vt" class:active={display === 'mesh'} onclick={() => (display = 'mesh')}>Mesh</button>
				<button class="vt" class:active={display === 'both'} onclick={() => (display = 'both')}>Both</button>
				<button class="vt" class:active={display === 'field'} disabled={field_results.length === 0} onclick={() => (display = 'field')}>Field</button>
				<button class="vt" class:active={display === 'plots'} onclick={() => (display = 'plots')}>Plots</button>
				{#if display === 'field' && field_results.length > 0}
					<div class="field-controls">
						<label>
							freq
							<select bind:value={field_freq_idx}>
								{#each field_results as r, i}
									<option value={i}>{(r.freq_hz / 1e9).toFixed(2)} GHz</option>
								{/each}
							</select>
						</label>
						<label>
							exc
							<select bind:value={field_exc_idx}>
								{#each field_results[0]?.fields ?? [] as _, i}
									<option value={i}>port {i + 1}</option>
								{/each}
							</select>
						</label>
					</div>
				{/if}
			</div>
			<div class="view-body">
				{#if display === 'plots'}
					<ResultsPanel {freqs} {smats} metrics={example.metrics} />
				{:else}
					<MeshViewer
						bind:this={viewer}
						mesh={mesh_data}
						mode={display === 'field' ? 'field' : display}
						field={display === 'field' ? active_field : null}
					/>
				{/if}
			</div>
		</main>
	</div>
</div>

<style>
	.app {
		display: flex;
		flex-direction: column;
		height: 100vh;
		background: var(--bg);
	}
	header {
		display: flex;
		align-items: center;
		padding: 0 16px;
		height: 36px;
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		flex-shrink: 0;
		gap: 12px;
	}
	.brand {
		text-decoration: none;
		display: flex;
		align-items: center;
	}
	.brand-text {
		font-family: var(--font-mono);
		font-size: var(--fs-md);
		font-weight: 600;
		color: var(--text);
		letter-spacing: -0.01em;
	}
	.tabs {
		display: flex;
		gap: 0;
		height: 100%;
	}
	.tab {
		display: flex;
		align-items: center;
		padding: 0 14px;
		font-size: var(--fs-xs);
		font-weight: 600;
		font-family: var(--font-mono);
		letter-spacing: 0.5px;
		color: var(--text-dim);
		text-decoration: none;
		text-transform: uppercase;
		transition: color var(--transition);
	}
	.tab:hover { color: var(--text-muted); }
	.tab.active { color: var(--accent); }
	.nav-sep {
		width: 1px;
		height: 100%;
		background: var(--border);
		flex-shrink: 0;
	}

	.body {
		display: flex;
		flex: 1;
		min-height: 0;
	}
	.sidebar {
		flex-shrink: 0;
		min-height: 0;
		display: flex;
		flex-direction: column;
		background: var(--bg);
	}
	.resize-handle-v {
		width: 2px;
		cursor: col-resize;
		background: var(--border);
		flex-shrink: 0;
		transition: background var(--transition);
	}
	.resize-handle-v:hover, .resize-handle-v:active { background: var(--accent); }
	.results-area {
		flex: 1;
		min-width: 0;
		min-height: 0;
		background: var(--bg);
		display: flex;
		flex-direction: column;
	}
	.view-tabs {
		display: flex;
		gap: 0;
		height: 32px;
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		flex-shrink: 0;
	}
	.vt {
		display: flex;
		align-items: center;
		padding: 0 14px;
		font-size: var(--fs-xs);
		font-weight: 600;
		font-family: var(--font-mono);
		letter-spacing: 0.5px;
		color: var(--text-dim);
		text-transform: uppercase;
		background: transparent;
		border: 0;
		border-right: 1px solid var(--border);
		cursor: pointer;
		transition: color var(--transition);
	}
	.vt:hover { color: var(--text-muted); }
	.vt:disabled { color: var(--text-dim); cursor: not-allowed; }
	.vt:disabled:hover { color: var(--text-dim); }
	.vt.active {
		color: var(--accent);
	}
	.field-controls {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-left: auto;
		padding-right: 12px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
	}
	.field-controls label {
		display: flex;
		align-items: center;
		gap: 4px;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.field-controls select {
		background: var(--bg-inset);
		border: 1px solid var(--input-border);
		color: var(--text);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		padding: 2px 4px;
	}
	.view-body {
		flex: 1;
		min-height: 0;
	}
	.desc {
		color: var(--text-muted);
		font-size: var(--fs-xs);
		line-height: 1.45;
		padding-top: 6px;
		font-family: var(--font-body);
	}
	.freq-list {
		display: flex;
		flex-wrap: wrap;
		gap: 3px;
	}
	.freq-chip {
		background: var(--bg-inset);
		border: 1px solid var(--border-subtle);
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		padding: 2px 6px;
	}
</style>
