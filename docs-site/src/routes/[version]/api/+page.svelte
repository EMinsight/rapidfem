<script lang="ts">
	import { DocLayout } from '$lib/components/layout';
	import { ModuleDoc } from '$lib/components/api';
	import { apiModules, site } from '$lib/config/rapidfem';
	import type { APIModule } from '$lib/api/types';
	import type { PageData } from './$types';

	let { data }: { data: PageData } = $props();

	// Resolved tag for display ('latest' → its concrete version).
	let resolvedTag = $derived(
		data.version === 'latest' ? data.manifest.latest : data.version
	);

	// Order modules by the curated config order, then any extras alphabetically.
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
	<title>API Reference {resolvedTag} — RapidFEM</title>
</svelte:head>

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

<style>
	.api-page-header {
		margin-bottom: var(--space-xl);
	}

	.api-title-row {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		margin-bottom: var(--space-sm);
	}

	.api-title-row h1 {
		margin: 0;
	}

	.lead {
		margin-bottom: 0;
	}
</style>
