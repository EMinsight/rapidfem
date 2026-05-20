<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { plotColors, fonts } from '$lib/theme';
	import type { TdTimeSeriesPayload, TdSeries } from '$lib/api';

	// Line-plot panel for the ProblemTD time-series results — driven_transient
	// probe signals (`domain:'time'`) and the scalar transfer_function
	// (`domain:'freq'`, plotted as magnitude in dB + phase). Built on the same
	// Plotly + theme machinery as ResultsPanel.

	let { payload = null }: { payload?: TdTimeSeriesPayload | null } = $props();

	let container = $state<HTMLDivElement | null>(null);
	let Plotly: any = $state(null);

	onMount(async () => {
		Plotly = (await import('plotly.js-dist-min')).default;
	});

	$effect(() => {
		if (!container) return;
		const ro = new ResizeObserver(() => {
			if (!Plotly) return;
			for (const div of container!.querySelectorAll('.plot-cell')) {
				Plotly.Plots?.resize(div as HTMLDivElement);
			}
		});
		ro.observe(container);
		for (const c of container.querySelectorAll('.plot-cell')) ro.observe(c);
		const onWindowResize = () => {
			if (!Plotly) return;
			for (const c of container!.querySelectorAll('.plot-cell')) {
				Plotly.Plots?.resize(c as HTMLDivElement);
			}
		};
		window.addEventListener('resize', onWindowResize);
		return () => { ro.disconnect(); window.removeEventListener('resize', onWindowResize); };
	});

	$effect(() => {
		const p = payload;
		const P = Plotly;
		const el = container;
		if (!P || !p || p.x.length === 0) return;
		if (!el) {
			tick().then(() => container && render(container, p, P));
			return;
		}
		render(el, p, P);
	});

	function mag_db(s: TdSeries): number[] {
		const re = s.y_re ?? [];
		const im = s.y_im ?? [];
		return re.map((r, k) => 20 * Math.log10(Math.max(1e-15, Math.hypot(r, im[k] ?? 0))));
	}
	function phase_deg(s: TdSeries): number[] {
		const re = s.y_re ?? [];
		const im = s.y_im ?? [];
		return re.map((r, k) => (Math.atan2(im[k] ?? 0, r) * 180) / Math.PI);
	}

	function render(el: HTMLDivElement, p: TdTimeSeriesPayload, P: any) {
		const base = {
			font: { family: fonts.mono, size: 10, color: plotColors.text },
			paper_bgcolor: 'rgba(0,0,0,0)',
			plot_bgcolor: plotColors.bg,
			margin: { t: 8, r: 12, b: 44, l: 60 },
			xaxis: {
				title: { text: p.x_label, font: { size: 10 }, standoff: 8 },
				gridcolor: plotColors.grid,
				linecolor: plotColors.axis,
				tickfont: { size: 10 }
			},
			autosize: true,
			showlegend: true,
			legend: {
				orientation: 'h', x: 0, xanchor: 'left', y: 1.02, yanchor: 'bottom',
				font: { size: 10, color: plotColors.text }, bgcolor: 'rgba(0,0,0,0)'
			}
		};
		const cfg = { responsive: true, displayModeBar: false };
		const tr = (y: number[], ci: number, name: string) => ({
			x: p.x, y, type: 'scatter' as const, mode: 'lines' as const,
			line: { color: plotColors.cycle[ci % plotColors.cycle.length], width: 2 },
			name, showlegend: true
		});
		const yax = (title: string) => ({
			title: { text: title, font: { size: 10 }, standoff: 12 },
			gridcolor: plotColors.grid, tickfont: { size: 10 }
		});

		const plots: Record<string, { data: any[]; yaxis: any } | null> = {
			'ts-main': null, 'ts-mag': null, 'ts-phase': null
		};

		if (p.domain === 'freq') {
			plots['ts-mag'] = {
				data: p.series.map((s, i) => tr(mag_db(s), i, `|${s.label}|`)),
				yaxis: yax('|H| (dB)')
			};
			plots['ts-phase'] = {
				data: p.series.map((s, i) => tr(phase_deg(s), i, `∠${s.label}`)),
				yaxis: yax('Phase (°)')
			};
		} else {
			plots['ts-main'] = {
				data: p.series.map((s, i) => tr(s.y ?? [], i, s.label)),
				yaxis: yax('Amplitude')
			};
		}

		for (const id of ['ts-main', 'ts-mag', 'ts-phase']) {
			const div = el.querySelector(`#${id}`) as HTMLDivElement | null;
			if (!div) continue;
			const plot = plots[id];
			if (plot) {
				P.react(div, plot.data, { ...base, yaxis: plot.yaxis, hovermode: 'closest' }, cfg);
			} else {
				P.purge(div);
			}
		}

		requestAnimationFrame(() => {
			for (const div of el.querySelectorAll('.plot-cell')) P.Plots?.resize(div);
		});
	}
</script>

{#if payload && payload.x.length > 0}
	<div class="results" bind:this={container}>
		{#if payload.source_label}
			<div class="source-tag">source · {payload.source_label}</div>
		{/if}
		<div class="plot-grid" class:hidden={payload.domain === 'freq'}>
			<div id="ts-main" class="plot-cell"></div>
		</div>
		<div class="plot-grid" class:hidden={payload.domain !== 'freq'}>
			<div id="ts-mag" class="plot-cell"></div>
			<div id="ts-phase" class="plot-cell"></div>
		</div>
	</div>
{:else}
	<div class="no-result">Run to see results</div>
{/if}

<style>
	.results {
		height: 100%;
		background: var(--bg-surface);
		padding: 6px;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.source-tag {
		font-family: var(--font-mono);
		font-size: var(--fs-xs, 11px);
		color: var(--text-dim);
		flex: 0 0 auto;
	}
	.plot-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		flex: 1 1 0;
		min-height: 0;
	}
	.plot-grid.hidden {
		display: none;
	}
	.plot-cell {
		flex: 1 1 380px;
		min-height: 240px;
		min-width: 0;
		background: var(--bg-panel);
		border: 1px solid var(--border-subtle);
	}
	.no-result {
		padding: 24px;
		text-align: center;
		color: var(--text-dim);
		font-size: var(--fs-sm);
		font-family: var(--font-mono);
	}
</style>
