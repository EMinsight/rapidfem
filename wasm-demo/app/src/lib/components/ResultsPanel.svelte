<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { plotColors, fonts } from '$lib/theme';
	import type { SMatrix } from '$lib/wasm';
	import { L_eq_pH, find_srf, sToZ } from '$lib/sparams';

	let {
		freqs = [],
		smats = [],
		extract_l = false
	}: {
		freqs?: number[];
		smats?: SMatrix[];
		extract_l?: boolean;
	} = $props();

	let container = $state<HTMLDivElement | null>(null);
	let Plotly: any = $state(null);

	onMount(async () => {
		Plotly = (await import('plotly.js-dist-min')).default;
	});

	$effect(() => {
		if (!container) return;
		const ro = new ResizeObserver(() => {
			if (!Plotly) return;
			const plots = container!.querySelectorAll('.plot-grid > div');
			for (const div of plots) Plotly.Plots?.resize(div);
		});
		ro.observe(container);
		return () => ro.disconnect();
	});

	$effect(() => {
		const f = freqs;
		const s = smats;
		const ex = extract_l;
		const P = Plotly;
		const el = container;
		if (!P || f.length === 0) return;
		if (!el) {
			tick().then(() => container && render(container, f, s, ex, P));
			return;
		}
		render(el, f, s, ex, P);
	});

	function s_mag(s: { re: number; im: number }): number {
		return 20 * Math.log10(Math.max(1e-15, Math.hypot(s.re, s.im)));
	}
	function s_phase(s: { re: number; im: number }): number {
		return (Math.atan2(s.im, s.re) * 180) / Math.PI;
	}

	function render(el: HTMLDivElement, f: number[], s: SMatrix[], ex: boolean, P: any) {
		const fHz = f;
		const xType = (Math.max(...fHz) / Math.max(1, Math.min(...fHz))) > 50 ? 'log' : 'linear';

		const base = {
			font: { family: fonts.mono, size: 10, color: plotColors.text },
			paper_bgcolor: 'rgba(0,0,0,0)',
			plot_bgcolor: plotColors.bg,
			margin: { t: 8, r: 12, b: 44, l: 60 },
			xaxis: {
				type: xType as any,
				title: { text: 'Frequency (Hz)', font: { size: 10 }, standoff: 8 },
				gridcolor: plotColors.grid,
				linecolor: plotColors.axis,
				tickfont: { size: 10 }
			},
			autosize: true,
			showlegend: false
		};
		const cfg = { responsive: true, displayModeBar: false };
		const tr = (y: (number | null)[], ci: number, name?: string) => ({
			x: fHz, y, type: 'scatter' as const, mode: 'lines+markers' as const,
			line: { color: plotColors.cycle[ci % plotColors.cycle.length], width: 2 },
			marker: { color: plotColors.cycle[ci % plotColors.cycle.length], size: 5 },
			...(name ? { name, showlegend: true } : {})
		});

		const yax = (title: string, ...datasets: (number | null)[][]): any => {
			const axis: any = { title: { text: title, font: { size: 10 }, standoff: 12 }, gridcolor: plotColors.grid, tickfont: { size: 10 } };
			const all = datasets.flat().filter((v): v is number => typeof v === 'number' && isFinite(v));
			if (all.length > 0) {
				const min = Math.min(...all);
				const max = Math.max(...all);
				const range = max - min;
				const absMax = Math.max(Math.abs(min), Math.abs(max));
				if (absMax > 0 && range / absMax < 0.05) {
					const pad = Math.max(range * 0.2, absMax * 0.005);
					axis.range = [min - pad, max + pad];
				}
			}
			return axis;
		};

		// S-magnitude / phase across all combos
		const n = s[0]?.length ?? 0;
		const sMagTraces: { y: number[]; name: string; ci: number }[] = [];
		const sPhTraces: { y: number[]; name: string; ci: number }[] = [];
		let ci = 0;
		for (let i = 0; i < n; i++) {
			for (let j = 0; j < n; j++) {
				const name = `S${i + 1}${j + 1}`;
				sMagTraces.push({ y: s.map((m) => s_mag(m[i][j])), name: `|${name}|`, ci });
				sPhTraces.push({ y: s.map((m) => s_phase(m[i][j])), name: `∠${name}`, ci });
				ci++;
			}
		}

		const plots: { id: string; data: any[]; yaxis: any; layout?: any }[] = [];

		if (sMagTraces.length > 0) {
			const multi = sMagTraces.length > 1;
			plots.push({
				id: 'p-smag',
				data: sMagTraces.map((t) => tr(t.y, t.ci, multi ? t.name : undefined)),
				yaxis: yax('|S| (dB)', ...sMagTraces.map((t) => t.y))
			});
			plots.push({
				id: 'p-sph',
				data: sPhTraces.map((t) => tr(t.y, t.ci, multi ? t.name : undefined)),
				yaxis: yax('Phase (°)', ...sPhTraces.map((t) => t.y))
			});
		}

		// Equivalent inductance + SRF marker (only for RFIC examples)
		if (ex && n >= 2) {
			const Leq = fHz.map((freq, k) => L_eq_pH(s[k], freq));
			const valid = Leq.filter((v) => isFinite(v) && Math.abs(v) < 1e6).map(Math.abs);
			const cap = valid.length > 0 ? 5 * Math.max(...valid) : 1e4;
			const Lplot: (number | null)[] = Leq.map((v) => (!isFinite(v) || Math.abs(v) > cap ? null : v));
			const fSRF = find_srf(fHz, s);
			const shapes: any[] = [];
			const annotations: any[] = [];
			if (fSRF != null) {
				shapes.push({
					type: 'line', x0: fSRF, x1: fSRF, y0: 0, y1: 1, yref: 'paper',
					line: { color: plotColors.cycle[1], width: 1, dash: 'dash' }
				});
				annotations.push({
					x: fSRF, y: 1, yref: 'paper',
					text: `SRF ≈ ${(fSRF / 1e9).toFixed(1)} GHz`, showarrow: false,
					yshift: 10, xanchor: 'left', font: { color: plotColors.cycle[1], size: 10 }
				});
			}
			plots.push({
				id: 'p-leq',
				data: [tr(Lplot, 1)],
				yaxis: yax('L_eq (pH)', Lplot),
				layout: { shapes, annotations }
			});

			// Quality factor at port 1
			const Q = s.map((mat) => {
				const Z = sToZ(mat, 50);
				return Z[0][0].im / Z[0][0].re;
			});
			plots.push({
				id: 'p-q',
				data: [tr(Q, 2)],
				yaxis: yax('Q (port 1)', Q)
			});
		}

		const want = new Map(plots.map((p) => [p.id, p]));
		for (const id of ['p-smag', 'p-sph', 'p-leq', 'p-q']) {
			const div = el.querySelector(`#${id}`) as HTMLDivElement | null;
			if (!div) continue;
			const p = want.get(id);
			if (p) {
				P.react(div, p.data, { ...base, yaxis: p.yaxis, hovermode: 'closest', ...(p.layout || {}) }, cfg);
			} else {
				P.purge(div);
			}
		}

		requestAnimationFrame(() => {
			const divs = el.querySelectorAll('.plot-cell');
			for (const div of divs) P.Plots?.resize(div);
		});
	}
</script>

{#if freqs.length > 0}
	<div class="results" bind:this={container}>
		<div class="plot-grid">
			<div class="plot-col">
				<div id="p-smag" class="plot-cell"></div>
				<div id="p-sph" class="plot-cell"></div>
			</div>
			{#if extract_l}
				<div class="plot-col">
					<div id="p-leq" class="plot-cell"></div>
					<div id="p-q" class="plot-cell"></div>
				</div>
			{/if}
		</div>
	</div>
{:else}
	<div class="no-result">Run sweep to see results</div>
{/if}

<style>
	.results {
		height: 100%;
		background: var(--bg-surface);
		padding: 6px;
	}
	.plot-grid {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
		height: 100%;
	}
	.plot-col {
		flex: 1 1 380px;
		display: flex;
		flex-direction: column;
		gap: 6px;
		min-width: 0;
	}
	.plot-cell {
		flex: 1 1 0;
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
