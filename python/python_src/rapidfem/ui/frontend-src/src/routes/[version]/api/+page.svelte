<script lang="ts">
	import '$lib/docs/docs.css';
	import { base } from '$app/paths';
	import { DocLayout } from '$lib/docs/components/layout';
	import { ModuleDoc } from '$lib/docs/components/api';
	import { apiModules, site } from '$lib/docs/config/rapidfem';
	import type { APIModule } from '$lib/docs/api/types';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	let resolvedTag = $derived(
		data.version === 'latest' ? data.manifest.latest : data.version
	);

	// Order modules by the curated config order, extras alphabetically.
	let orderedModules = $derived.by(() => {
		const order = apiModules.map((m) => m.name);
		const all = Object.values(data.api.modules) as APIModule[];
		return all.slice().sort((a, b) => {
			const ia = order.indexOf(a.name);
			const ib = order.indexOf(b.name);
			if (ia !== -1 && ib !== -1) return ia - ib;
			if (ia !== -1) return -1;
			if (ib !== -1) return 1;
			return a.name.localeCompare(b.name);
		});
	});
</script>

<svelte:head>
	<title>RapidFEM API Reference {resolvedTag}</title>
</svelte:head>

<div class="api-route">
	<header class="app-header">
		<a class="brand" href="{base}/" aria-label="RapidFEM">
			<img src="{base}/favicon.svg" alt="RapidFEM" class="logo" />
		</a>
		<span class="nav-sep"></span>
		<nav class="tabs">
			<a class="tab" href="{base}/notebook">Notebook</a>
			<a class="tab active" href="{base}/{data.version}/api">API</a>
		</nav>
	</header>

	<div class="rfdocs api-content">
		<DocLayout api={data.api} manifest={data.manifest} currentVersion={data.version}>
			<article class="prose">
				<header class="api-page-header">
					<div class="api-title-row">
						<h1>API Reference</h1>
						<span class="badge accent">{resolvedTag}</span>
					</div>
					<p class="lead">
						{site.name} — {Object.keys(data.api.modules).length} modules, extracted from the
						Python docstrings. Click a class or function to expand it.
					</p>
				</header>

				{#each orderedModules as module (module.name)}
					<ModuleDoc {module} />
				{/each}
			</article>
		</DocLayout>
	</div>
</div>

<style>
	.api-route {
		height: 100vh;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	/* App header — chrome, uses the frontend's own tokens (outside .rfdocs). */
	.app-header {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		height: 36px;
		padding: 0 var(--space-xl);
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
		height: 100%;
	}

	.tab {
		display: flex;
		align-items: center;
		padding: 0 14px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 600;
		letter-spacing: 0.5px;
		color: var(--text-dim);
		text-decoration: none;
		transition: color var(--transition);
	}

	.tab:hover {
		color: var(--text-muted);
	}

	.tab.active {
		color: var(--accent);
	}

	.api-content {
		flex: 1;
		min-height: 0;
		display: flex;
	}
</style>
