<script lang="ts">
	import { onMount } from 'svelte';
	import { EXAMPLES, type DemoExample } from '$lib/examples';
	import { run_streaming_sweep, type SMatrix, type FrequencyResult } from '$lib/wasm';
	import {
		L_eq_pH,
		Q_factor,
		find_srf,
		abs_s,
		sToZ
	} from '$lib/sparams';
	import { palette } from '$lib/theme';

	let selected_id = $state('spiral');
	let example = $derived(EXAMPLES[selected_id]);
	let running = $state(false);
	let status = $state('idle');
	let progress = $state(0); // 0..1
	let log_lines = $state<string[]>([]);

	let freqs: number[] = $state([]);
	let smats: SMatrix[] = $state([]);

	let s_plot_el: HTMLDivElement;
	let l_plot_el: HTMLDivElement;
	let abort_controller: AbortController | null = null;

	let Plotly: any = null;

	onMount(async () => {
		Plotly = (await import('plotly.js-dist-min')).default;
	});

	function log(msg: string) {
		log_lines = [...log_lines, msg];
	}

	const plotly_dark_layout = {
		paper_bgcolor: palette.bgPanel,
		plot_bgcolor: palette.bgInset,
		font: { color: palette.text, family: 'Inter, sans-serif', size: 12 },
		margin: { t: 30, l: 70, r: 30, b: 50 },
		legend: { orientation: 'h', y: 1.12 },
		xaxis: { gridcolor: palette.borderSubtle, zerolinecolor: palette.border, color: palette.textMuted },
		yaxis: { gridcolor: palette.borderSubtle, zerolinecolor: palette.border, color: palette.textMuted }
	};

	function update_s_plot() {
		if (!Plotly || !s_plot_el) return;
		const x = freqs.map((f) => f / 1e9);
		const has_two = smats.length > 0 && smats[0].length >= 2;
		const traces: any[] = [
			{
				x,
				y: abs_s(smats, 0, 0),
				mode: 'lines+markers',
				name: '|S11|',
				line: { color: palette.accent, width: 2 },
				marker: { size: 6 }
			}
		];
		if (has_two) {
			traces.push({
				x,
				y: abs_s(smats, 1, 0),
				mode: 'lines+markers',
				name: '|S21|',
				line: { color: '#5aad78', width: 2 },
				marker: { size: 6 }
			});
		}
		Plotly.react(
			s_plot_el,
			traces,
			{
				...plotly_dark_layout,
				xaxis: { ...plotly_dark_layout.xaxis, title: { text: 'Frequency [GHz]' } },
				yaxis: { ...plotly_dark_layout.yaxis, title: { text: '|S|' }, range: [0, 1.05] }
			},
			{ displayModeBar: false, responsive: true }
		);
	}

	function update_l_plot() {
		if (!Plotly || !l_plot_el) return;
		if (!example.extract_l) {
			Plotly.purge(l_plot_el);
			return;
		}
		const x = freqs.map((f) => f / 1e9);
		const Leq = freqs.map((f, k) => L_eq_pH(smats[k], f));
		const valid = Leq.filter((v) => isFinite(v) && Math.abs(v) < 1e6).map(Math.abs);
		const cap = valid.length > 0 ? 5 * Math.max(...valid) : 1e4;
		const Lplot = Leq.map((v) => (!isFinite(v) || Math.abs(v) > cap ? null : v));
		const fSRF = find_srf(freqs, smats);
		const shapes: any[] = [];
		const annotations: any[] = [];
		if (fSRF != null) {
			shapes.push({
				type: 'line',
				x0: fSRF / 1e9,
				x1: fSRF / 1e9,
				y0: 0,
				y1: 1,
				yref: 'paper',
				line: { color: palette.accentSecondary, width: 1, dash: 'dash' }
			});
			annotations.push({
				x: fSRF / 1e9,
				y: 1,
				yref: 'paper',
				text: `SRF ≈ ${(fSRF / 1e9).toFixed(1)} GHz`,
				showarrow: false,
				yshift: 10,
				xanchor: 'left',
				font: { color: palette.accentSecondary }
			});
		}
		Plotly.react(
			l_plot_el,
			[
				{
					x,
					y: Lplot,
					mode: 'lines+markers',
					name: 'L_eq(f)',
					line: { color: palette.accentSecondary, width: 2 },
					marker: { size: 6 }
				}
			],
			{
				...plotly_dark_layout,
				xaxis: { ...plotly_dark_layout.xaxis, title: { text: 'Frequency [GHz]' } },
				yaxis: { ...plotly_dark_layout.yaxis, title: { text: 'L_eq [pH]' }, zeroline: true },
				shapes,
				annotations
			},
			{ displayModeBar: false, responsive: true }
		);
	}

	$effect(() => {
		// Re-render plots when smats/freqs change
		smats; freqs;
		update_s_plot();
		update_l_plot();
	});

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
			const ex: DemoExample = example;
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
				on_point: (k, total, point: FrequencyResult) => {
					freqs = [...freqs, point.freq_hz];
					smats = [...smats, point.S];
					progress = (k + 1) / total;
					const s11 = Math.hypot(point.S[0][0].re, point.S[0][0].im);
					const s21 =
						point.S.length >= 2
							? Math.hypot(point.S[1][0].re, point.S[1][0].im)
							: NaN;
					log(
						`  f=${(point.freq_hz / 1e9).toFixed(1).padStart(5)} GHz · |S11|=${s11.toFixed(3)}` +
							(isFinite(s21) ? ` · |S21|=${s21.toFixed(3)}` : '') +
							` · ${point.solve_time_s.toFixed(2)}s`
					);
				}
			});

			if (ex.extract_l && smats.length > 0) {
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

<div class="app">
	<aside class="sidebar">
		<header>
			<h1>rapidfem</h1>
			<div class="tagline">in-browser FEM</div>
		</header>

		<section>
			<label class="lbl" for="example">Example</label>
			<select id="example" bind:value={selected_id} disabled={running}>
				{#each Object.values(EXAMPLES) as ex}
					<option value={ex.id}>{ex.label}</option>
				{/each}
			</select>
			<div class="desc">{example.description}</div>
		</section>

		<section>
			{#if !running}
				<button class="primary" onclick={run}>Run sweep</button>
			{:else}
				<button class="secondary" onclick={abort}>Abort</button>
			{/if}
		</section>

		<section class="progress-section">
			<div class="status-row">
				<span class="status">{status}</span>
				<span class="progress-pct">{Math.round(progress * 100)}%</span>
			</div>
			<div class="progress-bar">
				<div class="progress-fill" style="width: {progress * 100}%"></div>
			</div>
		</section>

		<section class="log-section">
			<div class="lbl">Log</div>
			<pre class="log">{log_lines.join('\n') || '—'}</pre>
		</section>
	</aside>

	<main>
		<div class="plot-card">
			<div class="plot-title">S-parameters</div>
			<div bind:this={s_plot_el} class="plot"></div>
		</div>

		{#if example.extract_l}
			<div class="plot-card">
				<div class="plot-title">Equivalent series inductance L_eq(f) — SRF where L_eq diverges</div>
				<div bind:this={l_plot_el} class="plot"></div>
			</div>
		{/if}
	</main>
</div>

<style>
	.app {
		display: grid;
		grid-template-columns: 320px 1fr;
		min-height: 100vh;
		gap: 0;
	}
	.sidebar {
		background: var(--bg-surface);
		border-right: 1px solid var(--border);
		padding: var(--space-2xl) var(--space-xl);
		display: flex;
		flex-direction: column;
		gap: var(--space-xl);
		overflow-y: auto;
	}
	.sidebar header {
		display: flex;
		align-items: baseline;
		gap: var(--space-md);
	}
	.sidebar h1 {
		font-size: var(--fs-lg);
		font-weight: 600;
		letter-spacing: -0.01em;
		color: var(--text);
	}
	.tagline {
		color: var(--text-muted);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	section {
		display: flex;
		flex-direction: column;
		gap: var(--space-sm);
	}
	.lbl {
		color: var(--text-muted);
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}
	.desc {
		color: var(--text-muted);
		font-size: var(--fs-sm);
		line-height: 1.4;
		padding-top: var(--space-sm);
	}
	button {
		padding: var(--space-md) var(--space-xl);
		font-family: var(--font-body);
		font-size: var(--fs-md);
		font-weight: 500;
		border: 1px solid var(--border);
		cursor: pointer;
		transition: background var(--transition), border-color var(--transition), color var(--transition);
	}
	button.primary {
		background: var(--accent);
		color: var(--white);
		border-color: var(--accent);
	}
	button.primary:hover {
		background: var(--accent-hover);
		border-color: var(--accent-hover);
	}
	button.secondary {
		background: transparent;
		color: var(--text);
		border-color: var(--border);
	}
	button.secondary:hover {
		border-color: var(--input-hover);
	}
	button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.progress-section {
		gap: var(--space-md);
	}
	.status-row {
		display: flex;
		justify-content: space-between;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		color: var(--text-muted);
	}
	.progress-pct {
		color: var(--accent);
	}
	.progress-bar {
		height: 4px;
		background: var(--bg-inset);
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background: var(--accent);
		transition: width var(--transition) ease-out;
	}
	.log-section { flex: 1; min-height: 0; }
	.log-section .lbl { margin-bottom: var(--space-sm); }
	.log {
		flex: 1;
		background: var(--bg-inset);
		border: 1px solid var(--border-subtle);
		padding: var(--space-md);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text);
		overflow-y: auto;
		max-height: 300px;
		white-space: pre-wrap;
	}
	main {
		padding: var(--space-2xl);
		display: flex;
		flex-direction: column;
		gap: var(--space-xl);
		overflow-y: auto;
	}
	.plot-card {
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
	}
	.plot-title {
		padding: var(--space-md) var(--space-xl);
		border-bottom: 1px solid var(--border-subtle);
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		color: var(--text-muted);
	}
	.plot {
		height: 360px;
	}
</style>
