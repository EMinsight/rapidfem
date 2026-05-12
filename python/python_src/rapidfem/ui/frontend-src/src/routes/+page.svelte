<script lang="ts">
	import { onMount } from 'svelte';

	// Load the embed script so the <fem-viewer> custom element is defined
	// before its tags get parsed/upgraded in the card grid.
	onMount(() => {
		if (customElements.get('fem-viewer')) return;
		const script = document.createElement('script');
		script.src = `${import.meta.env.BASE_URL || '/'}embed/fem-viewer.js`.replace(/\/+/g, '/');
		document.head.appendChild(script);
	});

	const examples = [
		{
			name: 'wr90',
			label: 'WR-90 Waveguide',
			desc: 'Rectangular waveguide section, 21-pt sweep across the X-band TE₁₀ mode',
		},
		{
			name: 'wr90_pml',
			label: 'WR-90 + PML',
			desc: 'PML as a matched load — |S₁₁| at the numerical floor across the whole band',
		},
		{
			name: 'iris_filter',
			label: 'Iris Bandpass',
			desc: 'Three inductive irises in WR-90 form a 3rd-order Chebyshev passband near 10 GHz',
		},
		{
			name: 'patch_antenna',
			label: 'Patch Antenna',
			desc: 'Edge-fed 2.4 GHz patch on FR-4 with a 5-slab PML enclosure and far-field pattern',
		},
		{
			name: 'coax_step',
			label: 'Coax Step',
			desc: '50 → 75 ohm coaxial impedance discontinuity, native coax TEM ports',
		},
		{
			name: 'microstrip_line',
			label: 'Microstrip Z₀',
			desc: '50 ohm microstrip on RO4003C, narrowed to the λ_g/2 sweet spot for clean S-params',
		},
	];

	const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '');
	const demo_url = `${base}/demo`;
	const embed_test_url = `${base}/embed/test`;
</script>

<svelte:head>
	<title>RapidFEM &mdash; frequency-domain Maxwell FEM in Rust</title>
	<meta name="description" content="Open-source frequency-domain electromagnetic FEM solver. Nedelec-2 edge elements, complex-symmetric sparse linear algebra, Python API, browser notebook UI." />
</svelte:head>

<div class="page">
	<header>
		<a class="brand" href="/">
			<img src="{base}/favicon.svg" alt="RapidFEM" class="logo" />
		</a>
		<span class="nav-sep"></span>
		<nav class="tabs">
			<a class="tab" href={demo_url}>Notebook</a>
			<a class="tab" href={embed_test_url}>Embed</a>
		</nav>
	</header>
	<div class="landing">
		<div class="hero">
			<h1>RapidFEM</h1>
			<p>Frequency-domain electromagnetic FEM in Rust. Nédélec-2 edge elements, complex-symmetric sparse solvers, Python API, browser notebook UI.</p>
		</div>
		<div class="cards">
			{#each examples as ex}
				<a class="card" href={`${demo_url}?example=${ex.name}`}>
					<div class="card-preview">
						{@html `<fem-viewer src="${base}/demo/${ex.name}.json" rotate cycle speed="0.6" width="100%" height="200px"></fem-viewer>`}
					</div>
					<div class="card-info">
						<h3>{ex.label}</h3>
						<p>{ex.desc}</p>
					</div>
				</a>
			{/each}
		</div>
		<a class="embed-hint" href={embed_test_url}>
			<span class="embed-tag">&lt;fem-viewer&gt;</span>
			<span>Embed FEM results on your website</span>
		</a>
	</div>
	<footer class="landing-footer">
		<a href="https://github.com/milanofthe/rapidfem" target="_blank" rel="noopener">GitHub</a>
		<span class="sep">/</span>
		<a href="https://pypi.org/project/rapidfem/" target="_blank" rel="noopener">PyPI</a>
		<span class="sep">/</span>
		<a href={demo_url}>Notebook</a>
		<span class="sep">/</span>
		<a href="https://rapidpassives.org" target="_blank" rel="noopener">RapidPassives</a>
		<span class="sep">/</span>
		<a href="https://milanrother.com" target="_blank" rel="noopener">Milan Rother</a>
	</footer>
</div>

<style>
	/* Identical palette + scale to the notebook UI — global vars from
	 * app.css carry through. */
	.page {
		background: var(--bg);
		min-height: 100vh;
		display: flex;
		flex-direction: column;
	}

	/* ── Header (matches /demo notebook header) ──────────────────────── */
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
	.brand {
		display: inline-flex;
		align-items: center;
		text-decoration: none;
	}
	.logo {
		height: 22px;
		width: auto;
		display: block;
	}
	.nav-sep {
		width: 1px;
		height: 100%;
		background: var(--border);
		flex-shrink: 0;
	}
	.tabs {
		display: flex;
		gap: 0;
		height: 100%;
	}
	header .tab {
		display: flex;
		align-items: center;
		padding: 0 14px;
		font-size: var(--fs-xs);
		font-weight: 600;
		font-family: var(--font-mono);
		letter-spacing: 0.5px;
		color: var(--text-dim);
		text-decoration: none;
		transition: color var(--transition);
	}
	header .tab:hover { color: var(--text-muted); }
	header .tab.active { color: var(--accent); }
	.landing {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 32px;
		padding: 40px;
		overflow-y: auto;
	}
	.hero {
		text-align: center;
		margin-bottom: 16px;
	}
	.hero h1 {
		font-size: 28px;
		font-weight: 700;
		color: var(--accent);
		font-family: var(--font-mono);
		letter-spacing: 2px;
		margin-bottom: 10px;
	}
	.hero p {
		font-size: var(--fs-sm);
		color: var(--text-muted);
		max-width: 480px;
		font-family: var(--font-mono);
		line-height: 1.5;
	}
	.cards {
		display: flex;
		gap: 16px;
		flex-wrap: wrap;
		justify-content: center;
		max-width: 900px;
	}
	.card {
		width: 200px;
		background: var(--bg-surface);
		border: 1px solid var(--border-subtle);
		text-decoration: none;
		color: inherit;
		transition: border-color var(--transition), transform var(--transition);
		display: flex;
		flex-direction: column;
	}
	.card:hover {
		border-color: var(--accent);
		transform: translateY(-2px);
	}
	.card-preview {
		width: 100%;
		overflow: hidden;
	}
	.card-info {
		padding: 12px 14px;
		border-top: 1px solid var(--border-subtle);
	}
	.card-info h3 {
		font-size: var(--fs-sm);
		font-weight: 600;
		color: var(--accent);
		font-family: var(--font-mono);
		margin-bottom: 4px;
	}
	.card-info p {
		font-size: var(--fs-xs);
		color: var(--text-dim);
		line-height: 1.4;
		font-family: var(--font-mono);
	}
	.embed-hint {
		display: flex;
		align-items: center;
		gap: 10px;
		text-decoration: none;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-dim);
		transition: color var(--transition);
	}
	.embed-hint:hover { color: var(--accent); }
	.embed-tag {
		font-weight: 600;
		color: var(--text-muted);
		border: 1px solid var(--border);
		padding: 3px 8px;
		transition: border-color var(--transition), color var(--transition);
	}
	.embed-hint:hover .embed-tag {
		border-color: var(--accent);
		color: var(--accent);
	}
	.landing-footer {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		padding: 0 16px;
		height: 36px;
		background: var(--bg-surface);
		border-top: 1px solid var(--border);
		flex-shrink: 0;
	}
	.sep {
		color: var(--border);
		font-size: var(--fs-xs);
	}
	.landing-footer a {
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-dim);
		text-decoration: none;
		letter-spacing: 0.3px;
		transition: color var(--transition);
	}
	.landing-footer a:hover { color: var(--accent); }
</style>
