<script lang="ts">
	import '$lib/components/fields.css';
	import { EXAMPLES, type DemoExample } from '$lib/examples';
	import { run_streaming_sweep, type SMatrix } from '$lib/wasm';
	import ParamSidebar from '$lib/components/ParamSidebar.svelte';
	import ResultsPanel from '$lib/components/ResultsPanel.svelte';
	import StatusPanel from '$lib/components/StatusPanel.svelte';
	import ExampleSelect from '$lib/components/ExampleSelect.svelte';
	import { L_eq_pH, Q_factor, find_srf } from '$lib/sparams';

	let selected_id = $state('spiral');
	let example = $derived<DemoExample>(EXAMPLES[selected_id]);

	let running = $state(false);
	let status = $state('idle');
	let progress = $state(0);
	let log_lines = $state<string[]>([]);

	let freqs = $state<number[]>([]);
	let smats = $state<SMatrix[]>([]);

	let abort_controller: AbortController | null = null;

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
		abort_controller = new AbortController();

		try {
			const ex = example;
			const [mesh_resp, toml_resp] = await Promise.all([
				fetch(ex.msh_url),
				fetch(ex.toml_url)
			]);
			const mesh_bytes = new Uint8Array(await mesh_resp.arrayBuffer());
			const config_toml = await toml_resp.text();
			log(`[${ex.label}] mesh ${(mesh_bytes.byteLength / 1024).toFixed(0)} KB · ${ex.frequencies_hz.length} freqs`);

			await run_streaming_sweep({
				mesh_bytes,
				config_toml,
				frequencies_hz: ex.frequencies_hz,
				abort_signal: abort_controller.signal,
				on_status: (m) => (status = m),
				on_point: (k, total, point) => {
					freqs = [...freqs, point.freq_hz];
					smats = [...smats, point.S];
					progress = (k + 1) / total;
					const s11 = Math.hypot(point.S[0][0].re, point.S[0][0].im);
					const s21 = point.S.length >= 2
						? Math.hypot(point.S[1][0].re, point.S[1][0].im)
						: NaN;
					log(
						`  f=${(point.freq_hz / 1e9).toFixed(1).padStart(5)} GHz · |S11|=${s11.toFixed(3)}` +
						(isFinite(s21) ? ` · |S21|=${s21.toFixed(3)}` : '') +
						` · ${point.solve_time_s.toFixed(2)}s`
					);
				}
			});

			if (example.extract_l && smats.length > 0) {
				const L0 = L_eq_pH(smats[0], freqs[0]);
				const Q0 = Q_factor(smats[0]);
				log(`L_eq(${(freqs[0] / 1e9).toFixed(1)} GHz) = ${L0.toFixed(1)} pH · Q = ${Q0.toFixed(1)}`);
				const fSRF = find_srf(freqs, smats);
				if (fSRF != null) log(`SRF ≈ ${(fSRF / 1e9).toFixed(1)} GHz (where L_eq diverges)`);
				else log(`no SRF in sweep range`);
			}
			status = 'done';
		} catch (e) {
			status = 'failed';
			log(`ERROR: ${e}`);
			console.error(e);
		} finally {
			running = false;
			abort_controller = null;
		}
	}

	function abort() {
		abort_controller?.abort();
		status = 'aborted';
		running = false;
	}
</script>

<svelte:head>
	<title>rapidfem — in-browser FEM</title>
	<meta name="description" content="WebAssembly-powered FEM EM solver. Solve Sky130 microstrip and spiral inductor S-parameters and L_eq(f) in your browser, with no backend." />
</svelte:head>

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
		<aside class="sidebar">
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

		<main class="results-area">
			<ResultsPanel {freqs} {smats} extract_l={example.extract_l} />
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
		display: grid;
		grid-template-columns: 280px 1fr;
		flex: 1;
		min-height: 0;
	}
	.sidebar {
		border-right: 1px solid var(--border);
		min-height: 0;
	}
	.results-area {
		min-width: 0;
		min-height: 0;
		background: var(--bg);
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
