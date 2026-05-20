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
	<main>
		<header class="hero">
			<img src="{base}/favicon.svg" alt="RapidFEM" class="hero-logo" />
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
				<a href={external.github} class="hero-btn">
					<Icon name="github" size={14} />
					GitHub
				</a>
			</div>
		</header>
	</main>

	<div class="separator"></div>

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

	<div class="separator"></div>

	<main>
		<section>
			<h2>Quick Start</h2>
			<p class="section-intro">{quickstart.description}</p>
			<CodeBlock code={quickstart.code} title={quickstart.title} lang="python" />
		</section>
	</main>
</div>

<style>
	.page-wrapper {
		flex: 1;
		overflow-x: hidden;
	}

	main {
		max-width: 1140px;
		margin: 0 auto;
		padding: 0 var(--space-lg);
	}

	.hero {
		text-align: center;
		padding: var(--space-4xl) 0 var(--space-3xl);
	}

	.hero-logo {
		height: 72px;
		width: auto;
		margin-bottom: var(--space-lg);
	}

	.hero-title {
		font-family: var(--font-mono);
		font-size: 38px;
		font-weight: 700;
		letter-spacing: 2px;
		color: var(--accent);
		line-height: 1;
		margin-bottom: var(--space-md);
	}

	.description {
		font-size: var(--fs-sm);
		color: var(--text-muted);
		max-width: 620px;
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
		padding: var(--space-2xl) 0;
	}

	h2 {
		font-size: var(--fs-xs);
		font-weight: 600;
		color: var(--accent);
		text-transform: uppercase;
		letter-spacing: 1px;
		margin-bottom: var(--space-lg);
	}

	.section-intro {
		color: var(--text-muted);
		max-width: var(--content-max-width);
		margin-bottom: var(--space-lg);
		line-height: 1.7;
	}

	@media (max-width: 600px) {
		.hero-title {
			font-size: 30px;
		}
		.hero-logo {
			height: 56px;
		}
		main {
			padding: 0 var(--space-md);
		}
	}
</style>
