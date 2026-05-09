<script lang="ts">
	import '$lib/components/fields.css';
	import { EXAMPLES, type DemoExample } from '$lib/examples';
	import { run_streaming_sweep, type SMatrix } from '$lib/wasm';
	import ParamSidebar from '$lib/components/ParamSidebar.svelte';
	import ResultsPanel from '$lib/components/ResultsPanel.svelte';
	import StatusPanel from '$lib/components/StatusPanel.svelte';
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
	<header class="topbar">
		<div class="brand">
			<span class="brand-name">rapidfem</span>
			<span class="brand-sep">·</span>
			<span class="brand-tag">in-browser FEM</span>
		</div>
		<div class="ext-link">
			<a href="https://github.com/milanofthe/rapidfem" target="_blank" rel="noopener">GitHub</a>
		</div>
	</header>

	<div class="body">
		<aside class="sidebar">
			<ParamSidebar>
				<div class="param-section">
					<h4>Example</h4>
					<div class="f">
						<span>Demo</span>
						<div class="fi">
							<select bind:value={selected_id} disabled={running}>
								{#each Object.values(EXAMPLES) as ex}
									<option value={ex.id}>{ex.label}</option>
								{/each}
							</select>
						</div>
					</div>
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
	.topbar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 8px 16px;
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border);
		height: 40px;
		flex-shrink: 0;
	}
	.brand {
		display: flex;
		align-items: baseline;
		gap: 8px;
	}
	.brand-name {
		font-family: var(--font-mono);
		font-size: var(--fs-md);
		font-weight: 600;
		color: var(--text);
	}
	.brand-sep { color: var(--text-dim); }
	.brand-tag {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 1px;
	}
	.ext-link a {
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		text-decoration: none;
		text-transform: uppercase;
		letter-spacing: 1px;
		transition: color var(--transition);
	}
	.ext-link a:hover { color: var(--accent); }

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
