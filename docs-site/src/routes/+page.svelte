<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '$lib/components/common/Icon.svelte';
	import { tooltip } from '$lib/components/common/Tooltip.svelte';
	import CodeBlock from '$lib/components/common/CodeBlock.svelte';
	import { site, external, installation, quickstart } from '$lib/config/rapidfem';

	let copiedCmd = $state<string | null>(null);

	async function copyCommand(cmd: string) {
		try {
			await navigator.clipboard.writeText(cmd);
			copiedCmd = cmd;
			setTimeout(() => (copiedCmd = null), 2000);
		} catch {
			copiedCmd = null;
		}
	}
</script>

<svelte:head>
	<title>RapidFEM Documentation</title>
</svelte:head>

<div class="page-wrapper">
	<div class="page-scroll">
	<main>
		<header class="hero">
			<h1 class="hero-title">RapidFEM</h1>
			<p class="description">{site.description}</p>
			<div class="hero-actions">
				<a href="{base}/latest/api/" class="hero-btn primary">
					<Icon name="braces" size={14} />
					API Reference
				</a>
				<a href={external.demo} class="hero-btn">
					<Icon name="play" size={14} />
					Examples
				</a>
			</div>
		</header>
	</main>

	<main>
		<section>
			<h2>Installation</h2>
			<p class="section-intro">
				Wheels for Windows, Linux, and macOS are built via CI — the Rust core is compiled ahead
				of time, no toolchain required. Gmsh is pulled in automatically.
			</p>
			<div class="install-grid">
				{#each installation as opt}
					<button
						class="install-card"
						onclick={() => copyCommand(opt.command)}
						use:tooltip={copiedCmd === opt.command ? 'Copied!' : 'Click to copy'}
					>
						<div class="panel-header">
							<span>{opt.name}</span>
							<Icon name={copiedCmd === opt.command ? 'check' : 'copy'} size={14} />
						</div>
						<div class="install-body">
							<code>{opt.command}</code>
						</div>
					</button>
				{/each}
			</div>
		</section>
	</main>

	<main>
		<section>
			<h2>Quick Start</h2>
			<p class="section-intro">{quickstart.description}</p>
			<CodeBlock code={quickstart.code} title={quickstart.title} lang="python" />
		</section>
	</main>
	</div>

	<footer class="site-footer">
		<a href={external.github} target="_blank" rel="noopener">GitHub</a>
		<span class="sep">/</span>
		<a href={external.pypi} target="_blank" rel="noopener">PyPI</a>
		<span class="sep">/</span>
		<a href={external.demo} target="_blank" rel="noopener">Examples</a>
		<span class="sep">/</span>
		<a href="https://rapidpassives.org" target="_blank" rel="noopener">RapidPassives</a>
		<span class="sep">/</span>
		<a href="https://milanrother.com" target="_blank" rel="noopener">Milan Rother</a>
	</footer>
</div>

<style>
	.page-wrapper {
		flex: 1;
		min-height: 0;
		display: flex;
		flex-direction: column;
	}

	/* Scrolls; the footer below stays pinned to the viewport bottom. */
	.page-scroll {
		flex: 1;
		min-height: 0;
		overflow-y: auto;
		overflow-x: hidden;
	}

	main {
		width: 100%;
		max-width: 1140px;
		margin: 0 auto;
		padding: 0 var(--space-lg);
	}

	.hero {
		text-align: center;
		padding: var(--space-4xl) 0 var(--space-3xl);
	}

	.hero-title {
		font-family: var(--font-mono);
		font-size: 28px;
		font-weight: 700;
		letter-spacing: 2px;
		color: var(--accent);
		line-height: 1;
		margin-bottom: var(--space-md);
	}

	.description {
		font-size: var(--fs-sm);
		color: var(--text-muted);
		max-width: 760px;
		margin: 0 auto var(--space-xl);
		line-height: 1.7;
	}

	.hero-actions {
		display: flex;
		justify-content: center;
		flex-wrap: wrap;
		gap: var(--space-sm);
	}

	.hero-btn {
		display: inline-flex;
		align-items: center;
		gap: var(--space-sm);
		padding: var(--space-sm) var(--space-lg);
		border: 1px solid var(--border);
		background: var(--surface-raised);
		color: var(--text);
		font-size: var(--fs-xs);
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		text-decoration: none;
		transition: all var(--transition-fast);
	}

	.hero-btn:hover {
		border-color: var(--border-focus);
		background: var(--surface-hover);
		text-decoration: none;
	}

	.hero-btn.primary {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--surface);
	}

	.hero-btn.primary:hover {
		background: var(--accent-hover);
		border-color: var(--accent-hover);
	}

	section {
		padding: var(--space-3xl) 0;
	}

	h2 {
		font-family: var(--font-mono);
		font-size: 22px;
		font-weight: 700;
		color: var(--accent);
		text-transform: uppercase;
		letter-spacing: 1.6px;
		text-align: center;
		margin-bottom: var(--space-lg);
	}

	.section-intro {
		font-size: var(--fs-sm);
		color: var(--text-muted);
		max-width: 880px;
		margin: 0 auto var(--space-xl);
		line-height: 1.7;
		text-align: center;
	}

	/* Footer — pinned to the viewport bottom, like the RapidFEM demo. */
	.site-footer {
		flex-shrink: 0;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-wrap: wrap;
		gap: var(--space-md);
		height: 36px;
		padding: 0 var(--space-lg);
		background: var(--surface-raised);
		border-top: 1px solid var(--border);
		flex-shrink: 0;
	}

	.site-footer a {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		letter-spacing: 0.3px;
		color: var(--text-disabled);
		text-decoration: none;
		transition: color var(--transition-fast);
	}

	.site-footer a:hover {
		color: var(--accent);
		text-decoration: none;
	}

	.site-footer .sep {
		color: var(--border);
		font-size: var(--fs-xs);
	}

	@media (max-width: 600px) {
		.hero-title {
			font-size: 22px;
		}
		main {
			padding: 0 var(--space-md);
		}
	}
</style>
