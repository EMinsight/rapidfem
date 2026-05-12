<script lang="ts">
	import { onMount } from 'svelte';
	import CodeSnippet from '$lib/components/CodeSnippet.svelte';

	onMount(() => {
		if (customElements.get('fem-viewer')) return;
		const script = document.createElement('script');
		script.src = '/embed/fem-viewer.js';
		document.head.appendChild(script);
	});

	const demo_root = '/demo';

	const examples = [
		{
			title: 'Default',
			src: 'wr90',
			attrs: '',
			desc: 'Static 3D view, no interaction. Useful for documentation thumbnails.',
		},
		{
			title: 'Rotating',
			src: 'wr90',
			attrs: 'rotate',
			desc: 'Continuous camera orbit at the default speed.',
		},
		{
			title: 'Cycle &mdash; Geometry / Mesh / Field',
			src: 'wr90',
			attrs: 'rotate cycle',
			desc: 'Animated walk through the three display modes, every ~2 seconds. Same trick the landing cards use.',
		},
		{
			title: 'Interactive',
			src: 'patch_antenna',
			attrs: 'interactive',
			desc: 'Orbit (left-drag), pan (right-drag), zoom (wheel), double-click to fit.',
		},
		{
			title: 'Field mode',
			src: 'iris_filter',
			attrs: 'rotate mode="field" field-mode="lin" field-freq="20" field-port="0"',
			desc: 'Static field display at a chosen frequency / port index.',
		},
		{
			title: 'Mesh mode',
			src: 'coax_step',
			attrs: 'rotate mode="mesh"',
			desc: 'Wireframe view of every named surface — useful for visually spotting bad mesh density.',
		},
		{
			title: 'Custom camera preset',
			src: 'microstrip_line',
			attrs: 'interactive theta="60" phi="20"',
			desc: 'Pin the initial view with theta / phi (degrees). Combine with interactive to let the user move from there.',
		},
		{
			title: 'Transparent background',
			src: 'wr90_pml',
			attrs: 'rotate transparent',
			desc: 'Blends with whatever the host page uses behind the canvas.',
			transparent: true,
		},
	];

	function buildSnippet(name: string, attrs: string): string {
		const a = attrs ? '\n  ' + attrs.split(' ').join('\n  ') : '';
		return `<script src="https://fem.rapidpassives.org/embed/fem-viewer.js"><\/script>\n<fem-viewer\n  src="https://fem.rapidpassives.org/demo/${name}.json"${a}\n  width="100%" height="300px"\n><\/fem-viewer>`;
	}
</script>

<svelte:head>
	<title>Embed Test &mdash; RapidFEM</title>
	<meta name="robots" content="noindex" />
</svelte:head>

<div class="page">
	<header>
		<a class="brand" href="/" data-sveltekit-reload><img src="/favicon.svg" alt="RapidFEM" /></a>
		<span class="nav-sep"></span>
		<nav class="tabs">
			<a class="tab" href="/demo" data-sveltekit-reload>Notebook</a>
			<a class="tab active" href="/embed/test">Embed</a>
		</nav>
	</header>

	<div class="content">
		<div class="intro">
			<h1>Embeddable FEM Viewer</h1>
			<p>Drop simulated antenna / waveguide / line results onto any page with a single script tag.</p>
		</div>

		{#each examples as ex}
			<div class="example">
				<div class="example-info">
					<h3>{@html ex.title}</h3>
					<p>{ex.desc}</p>
					<CodeSnippet code={buildSnippet(ex.src, ex.attrs)} />
				</div>
				<div class="example-preview" class:checkered={ex.transparent}>
					{@html `<fem-viewer src="${demo_root}/${ex.src}.json" ${ex.attrs} width="100%" height="300px"></fem-viewer>`}
				</div>
			</div>
		{/each}
	</div>
</div>

<style>
	.page {
		background: #1c1c21;        /* match the landing page's lifted bg */
		min-height: 100vh;
		display: flex;
		flex-direction: column;
	}
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
	header .brand {
		display: inline-flex;
		align-items: center;
		text-decoration: none;
	}
	header .brand img { height: 22px; display: block; }
	header .nav-sep {
		width: 1px;
		height: 100%;
		background: var(--border);
		flex-shrink: 0;
	}
	header .tabs {
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

	.content {
		padding: 40px 40px 80px;
		display: flex;
		flex-direction: column;
		gap: 40px;
		max-width: 1100px;
		margin: 0 auto;
		width: 100%;
	}
	.intro h1 {
		font-size: var(--fs-lg);
		font-family: var(--font-mono);
		color: var(--accent);
		margin-bottom: 6px;
		letter-spacing: 1px;
	}
	.intro p {
		font-size: var(--fs-sm);
		font-family: var(--font-mono);
		color: var(--text-dim);
	}
	.example {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 16px;
		border: 1px solid var(--border-subtle);
		background: var(--bg-surface);
		padding: 16px;
	}
	.example-info {
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.example-info h3 {
		font-size: var(--fs-sm);
		font-family: var(--font-mono);
		color: var(--accent);
		font-weight: 600;
	}
	.example-info p {
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-dim);
		line-height: 1.5;
	}
	.example-preview {
		min-height: 300px;
		overflow: hidden;
	}
	.checkered {
		background-image: linear-gradient(45deg, #222 25%, transparent 25%),
			linear-gradient(-45deg, #222 25%, transparent 25%),
			linear-gradient(45deg, transparent 75%, #222 75%),
			linear-gradient(-45deg, transparent 75%, #222 75%);
		background-size: 16px 16px;
		background-position: 0 0, 0 8px, 8px -8px, -8px 0;
		background-color: #1a1a1a;
	}
	@media (max-width: 800px) {
		.example { grid-template-columns: 1fr; }
	}
</style>
