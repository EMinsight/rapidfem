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
	const embed_test_url = `${base}/embed/test.html`;
</script>

<svelte:head>
	<title>rapidfem &mdash; frequency-domain Maxwell FEM in Rust</title>
	<meta name="description" content="Open-source frequency-domain electromagnetic FEM solver. Nedelec-2 edge elements, complex-symmetric sparse linear algebra, Python API, browser notebook UI." />
</svelte:head>

<div class="page">
	<div class="landing">
		<div class="hero">
			<img src="{base}/favicon.svg" alt="rapidfem" class="logo" />
			<h1>rapidfem</h1>
			<p>Frequency-domain electromagnetic FEM in Rust. Nédélec-2 edge elements, complex-symmetric sparse solvers, Python API, browser notebook UI.</p>
			<div class="cta">
				<a class="cta-primary" href={demo_url}>Open notebook demo &rarr;</a>
				<a class="cta-secondary" href="https://github.com/milanofthe/rapidfem" target="_blank" rel="noopener">GitHub</a>
			</div>
			<pre class="install">pip install rapidfem[ui]   <span class="comment"># solver + browser UI</span></pre>
		</div>

		<div class="cards">
			{#each examples as ex}
				<a class="card" href={`${demo_url}?example=${ex.name}`}>
					<div class="card-preview">
						{@html `<fem-viewer src="${base}/demo/${ex.name}.json" rotate animate-field speed="0.6" width="100%" height="200px"></fem-viewer>`}
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
		<a href="https://rapidpassives.org" target="_blank" rel="noopener">RapidPassives</a>
		<span class="sep">/</span>
		<a href="https://milanrother.com" target="_blank" rel="noopener">Milan Rother</a>
	</footer>
</div>

<style>
	.page {
		min-height: 100vh;
		display: flex;
		flex-direction: column;
	}
	.landing {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 32px;
		padding: 40px 24px;
	}

	/* ── Hero ─────────────────────────────────────────────────────────── */
	.hero {
		display: flex;
		flex-direction: column;
		align-items: center;
		text-align: center;
		gap: 12px;
		max-width: 640px;
	}
	.hero .logo {
		width: 56px;
		height: 56px;
		margin-bottom: 4px;
	}
	.hero h1 {
		font-size: 32px;
		font-weight: 700;
		color: var(--accent);
		font-family: var(--font-mono);
		letter-spacing: 2px;
		margin: 0;
	}
	.hero p {
		font-size: var(--fs-sm);
		color: var(--text-muted);
		font-family: var(--font-mono);
		line-height: 1.55;
		margin: 0;
	}
	.cta {
		display: flex;
		gap: 12px;
		margin-top: 4px;
	}
	.cta-primary, .cta-secondary {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		text-decoration: none;
		padding: 8px 16px;
		border: 1px solid var(--border);
		letter-spacing: 0.5px;
		transition: border-color var(--transition), color var(--transition), background var(--transition);
	}
	.cta-primary {
		background: var(--accent);
		color: var(--bg);
		border-color: var(--accent);
		font-weight: 600;
	}
	.cta-primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
	.cta-secondary { color: var(--text-muted); }
	.cta-secondary:hover { color: var(--accent); border-color: var(--accent); }
	.install {
		background: var(--bg-mid);
		border: 1px solid var(--border-subtle);
		padding: 10px 14px;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text);
		margin: 8px 0 0;
		text-align: left;
	}
	.install .comment { color: var(--text-dim); }

	/* ── Cards ────────────────────────────────────────────────────────── */
	.cards {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
		gap: 16px;
		width: 100%;
		max-width: 980px;
	}
	.card {
		background: var(--bg-mid);
		border: 1px solid var(--border-subtle);
		text-decoration: none;
		color: inherit;
		display: flex;
		flex-direction: column;
		transition: border-color var(--transition), transform var(--transition);
	}
	.card:hover {
		border-color: var(--accent);
		transform: translateY(-2px);
	}
	.card-preview {
		width: 100%;
		overflow: hidden;
		background: var(--canvas-bg);
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
		margin: 0 0 4px;
		letter-spacing: 0.3px;
	}
	.card-info p {
		font-size: var(--fs-xs);
		color: var(--text-dim);
		line-height: 1.45;
		font-family: var(--font-mono);
		margin: 0;
	}

	/* ── Embed hint ───────────────────────────────────────────────────── */
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

	/* ── Footer ───────────────────────────────────────────────────────── */
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
	.landing-footer a {
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-dim);
		text-decoration: none;
		letter-spacing: 0.3px;
		transition: color var(--transition);
	}
	.landing-footer a:hover { color: var(--accent); }
	.sep { color: var(--border); font-size: var(--fs-xs); }
</style>
