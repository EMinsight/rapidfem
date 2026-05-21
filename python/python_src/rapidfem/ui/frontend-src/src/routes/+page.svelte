<script lang="ts">
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import { IS_STATIC_MODE } from '$lib/static_mode';
	import { installation, quickstart } from '$lib/docs/config/rapidfem';
	import '$lib/docs/docs.css';
	import CodeBlock from '$lib/docs/components/common/CodeBlock.svelte';
	import Icon from '$lib/docs/components/common/Icon.svelte';

	// Quick-Start tab — index into the `quickstart` array (FD / TD).
	let qs_tab = $state(0);

	let copied_cmd = $state<string | null>(null);
	let cmd_timer: ReturnType<typeof setTimeout> | null = null;
	function copyCmd(cmd: string) {
		try {
			navigator.clipboard.writeText(cmd);
			copied_cmd = cmd;
			if (cmd_timer) clearTimeout(cmd_timer);
			cmd_timer = setTimeout(() => { copied_cmd = null; }, 1400);
		} catch {}
	}

	// Local `rapidfem serve` has no use for the landing page — go straight
	// to /notebook. The landing route is only meaningful in the GH-Pages
	// build (VITE_STATIC_MODE=1) where this is the fem.rapidpassives.org
	// entry point. At build time IS_STATIC_MODE is false in the local
	// serve bundle, so the {#if} below drops the landing DOM from the
	// prerendered HTML too — no flash of unwanted content.
	onMount(() => {
		if (!IS_STATIC_MODE) {
			const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '');
			goto(`${base}/notebook`, { replaceState: true });
			return;
		}
		// Static-demo only: define <fem-viewer> for the card grid.
		if (customElements.get('fem-viewer')) return;
		const script = document.createElement('script');
		script.src = `${import.meta.env.BASE_URL || '/'}embed/fem-viewer.js`.replace(/\/+/g, '/');
		document.head.appendChild(script);
	});

	// The curated frequency-domain demo cards — `name` is the baked-example
	// stem (kept in sync with DEMO_EXAMPLES in scripts/bake_demo.py). The
	// <fem-viewer> embed renders geometry / mesh / field, so only the
	// frequency-domain examples carry cards; the time-domain examples ship
	// in the demo too and open from the notebook.
	const examples = [
		{
			name: 'fd_wr90',
			label: 'WR-90 Waveguide',
			desc: 'Rectangular waveguide section, 21-pt sweep across the X-band TE₁₀ mode',
		},
		{
			name: 'fd_coax_step',
			label: 'Coax Step',
			desc: '50 → 75 ohm coaxial impedance discontinuity, native coax TEM ports',
			fieldFreq: 0,
		},
		{
			name: 'fd_microstrip_line',
			label: 'Microstrip Z₀',
			desc: '50 ohm microstrip on RO4003C, narrowed to the λ_g/2 sweet spot for clean S-params',
			fieldMode: 'log',
		},
		{
			name: 'fd_iris_filter',
			label: 'Iris Bandpass',
			desc: 'Three inductive irises in WR-90 form a 3rd-order Chebyshev passband near 10 GHz',
			fieldFreq: 24,
		},
		{
			name: 'fd_patch_antenna',
			label: 'Patch Antenna',
			desc: 'Edge-fed 2.4 GHz patch on FR-4 with a 5-slab PML enclosure and far-field pattern',
			fieldFreq: 0,
			fieldMode: 'log',
		},
		{
			name: 'fd_pyramidal_horn',
			label: 'Pyramidal Horn',
			desc: 'Flared rectangular-waveguide horn antenna with a PML radiation box and far-field pattern',
			fieldMode: 'log',
		},
	] as Array<{
		name: string; label: string; desc: string;
		fieldFreq?: number; fieldMode?: 'lin' | 'log';
	}>;

	const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '');
	const notebook_url = `${base}/notebook`;
	const api_url = `${base}/latest/api`;
	const embed_test_url = `${base}/embed/test`;

	const install_cmd = 'pip install rapidfem';
	let copied = false;
	let copy_timer: ReturnType<typeof setTimeout> | null = null;
	function copyInstall() {
		try {
			navigator.clipboard.writeText(install_cmd);
			copied = true;
			if (copy_timer) clearTimeout(copy_timer);
			copy_timer = setTimeout(() => { copied = false; }, 1400);
		} catch {}
	}
</script>

<svelte:head>
	<title>RapidFEM: electromagnetic Maxwell FEM in Rust</title>
	<meta name="description" content="Open-source electromagnetic FEM solver: a frequency-domain Nédélec edge-element backend and a time-domain DGTD backend. Rust core, Python API, browser notebook UI." />
</svelte:head>

{#if IS_STATIC_MODE}
<div class="page">
	<header>
		<a class="brand" href="/">
			<img src="{base}/favicon.svg" alt="RapidFEM" class="logo" />
		</a>
		<span class="nav-sep"></span>
		<nav class="tabs">
			<a class="tab" href={notebook_url}>Notebook</a>
			<a class="tab" href={api_url}>API</a>
			<a class="tab" href={embed_test_url}>Embed</a>
		</nav>
	</header>
	<div class="landing">
		<div class="hero">
			<div class="hero-head">
				<img src="{base}/favicon.svg" alt="" class="hero-icon" />
				<h1>RapidFEM</h1>
			</div>
			<p>Electromagnetic FEM in Rust: a frequency-domain Nédélec edge-element solver and a time-domain DGTD backend behind one geometry API. CFL-free exponential time stepping, a Python API, and a browser-based notebook.</p>
		</div>
		<div class="quickstart">
			<div class="install-line">
				<span class="install-prompt">$</span>
				<code class="install-cmd">{install_cmd}</code>
				<button class="copy-btn" onclick={copyInstall} aria-label="Copy install command">
					{copied ? '✓ copied' : 'copy'}
				</button>
			</div>
			<a class="cta" href={notebook_url}>
				<span>Open the Notebook</span>
				<span class="cta-arrow">→</span>
			</a>
		</div>
		<div class="cards">
			{#each examples as ex}
				<a class="card" href={`${notebook_url}?example=${ex.name}`}>
					<div class="card-preview">
						{@html `<fem-viewer src="${base}/demo/${ex.name}.json" rotate cycle speed="0.6" field-samples="5000"${ex.fieldFreq !== undefined ? ` field-freq="${ex.fieldFreq}"` : ''}${ex.fieldMode ? ` field-mode="${ex.fieldMode}"` : ''} width="100%" height="240px"></fem-viewer>`}
					</div>
					<div class="card-info">
						<h3>{ex.label}</h3>
						<p>{ex.desc}</p>
					</div>
				</a>
			{/each}
		</div>
		<section class="info-section">
			<h2 class="section-title">Installation</h2>
			<div class="rfdocs snippet-block">
				<div class="install-grid">
					{#each installation as opt}
						<button class="install-card" onclick={() => copyCmd(opt.command)}>
							<div class="panel-header">
								<span>{opt.name}</span>
								<Icon name={copied_cmd === opt.command ? 'check' : 'copy'} size={14} />
							</div>
							<div class="install-body"><code>{opt.command}</code></div>
						</button>
					{/each}
				</div>
			</div>
		</section>

		<section class="info-section">
			<h2 class="section-title">Quick Start</h2>
			<div class="qs-tabs">
				{#each quickstart as qs, i}
					<button class="qs-tab" class:active={qs_tab === i} onclick={() => (qs_tab = i)}>
						{qs.title}
					</button>
				{/each}
			</div>
			<p class="section-desc">{quickstart[qs_tab].description}</p>
			<div class="rfdocs snippet-block">
				<CodeBlock code={quickstart[qs_tab].code} title={quickstart[qs_tab].title} lang="python" />
			</div>
		</section>

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
		<a href={notebook_url}>Notebook</a>
		<span class="sep">/</span>
		<a href="https://rapidpassives.org" target="_blank" rel="noopener">RapidPassives</a>
		<span class="sep">/</span>
		<a href="https://milanrother.com" target="_blank" rel="noopener">Milan Rother</a>
	</footer>
</div>
{/if}

<style>
	/* Notebook's palette + scale (global vars from app.css) — but a
	 * lifted mid-gray bg so the tiles read as cards instead of floating
	 * voids on the viewer-dark default. Fixed viewport height with flex
	 * column so the header stays anchored at the top, the footer at the
	 * bottom, and `.landing` scrolls internally — matches rapidpassives. */
	.page {
		background: #1c1c21;
		height: 100vh;
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
		gap: 64px;
		padding: 48px 40px;
		overflow-y: auto;
		/* Shared content width — one row of the example grid: three 320px
		   cards plus two 18px gaps. The card grid, the Installation block
		   and the Quick Start snippet all align to it. */
		--content-w: 996px;
	}
	.hero {
		text-align: center;
		margin-bottom: 16px;
	}
	.hero-head {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 16px;
		margin-bottom: 24px;
	}
	.hero-icon {
		height: 44px;
		width: auto;
		display: block;
	}
	.hero h1 {
		font-size: 28px;
		font-weight: 700;
		color: var(--accent);
		font-family: var(--font-mono);
		letter-spacing: 2px;
		margin: 0;
	}
	.hero p {
		font-size: var(--fs-sm);
		color: var(--text-muted);
		max-width: 720px;
		font-family: var(--font-mono);
		line-height: 1.5;
	}
	.quickstart {
		display: flex;
		align-items: stretch;
		gap: 14px;
		flex-wrap: wrap;
		justify-content: center;
		margin-top: -8px;
	}
	.install-line {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		padding: 0 12px;
		height: 36px;
		background: var(--bg-surface);
		border: 1px solid var(--border-subtle);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.install-prompt {
		color: var(--text-dim);
		user-select: none;
	}
	.install-cmd {
		color: var(--text);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		letter-spacing: 0.3px;
	}
	.copy-btn {
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 10px;
		letter-spacing: 0.5px;
		padding: 3px 8px;
		cursor: pointer;
		text-transform: uppercase;
		transition: color var(--transition), border-color var(--transition);
	}
	.copy-btn:hover {
		color: var(--accent);
		border-color: var(--accent);
	}
	.cta {
		display: inline-flex;
		align-items: center;
		gap: 10px;
		padding: 0 18px;
		height: 36px;
		background: var(--accent);
		color: var(--bg);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 700;
		letter-spacing: 0.6px;
		text-decoration: none;
		text-transform: uppercase;
		transition: filter var(--transition), transform var(--transition);
	}
	.cta:hover {
		filter: brightness(1.1);
		transform: translateY(-1px);
	}
	.cta-arrow {
		font-size: 14px;
		line-height: 1;
		transition: transform var(--transition);
	}
	.cta:hover .cta-arrow { transform: translateX(2px); }
	.quickstart-note {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-dim);
		text-align: center;
		max-width: 560px;
		line-height: 1.6;
		margin-top: -12px;
	}
	.quickstart-note a {
		color: var(--text-muted);
		text-decoration: none;
		border-bottom: 1px solid var(--border);
		transition: color var(--transition), border-color var(--transition);
	}
	.quickstart-note a:hover {
		color: var(--accent);
		border-bottom-color: var(--accent);
	}
	.cards {
		display: flex;
		gap: 18px;
		flex-wrap: wrap;
		justify-content: center;
		max-width: var(--content-w);
	}
	.card {
		width: 320px;
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
	/* ── Installation + Quick Start sections ─────────────────────────── */
	.info-section {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 14px;
		width: 100%;
		max-width: var(--content-w);
	}
	.section-title {
		font-family: var(--font-mono);
		font-size: 22px;
		font-weight: 700;
		color: var(--accent);
		text-transform: uppercase;
		letter-spacing: 1.6px;
	}
	.section-desc {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		color: var(--text-muted);
		line-height: 1.6;
		text-align: center;
		max-width: 720px;
		/* balance line fill so the wrap never leaves a lone word */
		text-wrap: balance;
	}
	/* Frequency-domain / time-domain switch for the Quick Start snippet. */
	.qs-tabs {
		display: flex;
		gap: 4px;
	}
	.qs-tab {
		font-family: var(--font-mono);
		font-size: var(--fs-xs, 11px);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		padding: 5px 12px;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		color: var(--text-muted);
		cursor: pointer;
		transition: background var(--transition), border-color var(--transition), color var(--transition);
	}
	.qs-tab:hover { color: var(--text); border-color: var(--accent); }
	.qs-tab.active {
		color: var(--accent);
		border-color: var(--accent);
		background: var(--accent-dim);
	}
	/* Wrapper for the ported docs snippets (install cards + CodeBlock).
	   .rfdocs supplies the docs design tokens; docs.css does the styling. */
	.snippet-block {
		width: 100%;
		max-width: var(--content-w);
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
