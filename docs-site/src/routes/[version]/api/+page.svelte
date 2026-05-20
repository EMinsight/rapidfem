<script lang="ts">
	import { page } from '$app/stores';
	import { apiModules } from '$lib/config/rapidfem';

	let version = $derived($page.params.version);
</script>

<svelte:head>
	<title>API Reference — RapidFEM</title>
</svelte:head>

<div class="api-stub">
	<h1>API Reference</h1>
	<p class="lead">
		Version <code>{version}</code> — generated from the RapidFEM Python docstrings.
	</p>
	<p class="placeholder">
		The extracted API content is wired up in a later step. Modules to be documented:
	</p>
	<ul class="module-list">
		{#each apiModules as mod}
			<li><code>{mod.name}</code> — {mod.description}</li>
		{/each}
	</ul>
</div>

<style>
	.api-stub {
		max-width: var(--content-max-width);
		margin: 0 auto;
		padding: var(--space-2xl) var(--space-lg);
	}

	.placeholder {
		color: var(--text-muted);
		margin-top: var(--space-xl);
	}

	.module-list {
		list-style: none;
		padding: 0;
		margin-top: var(--space-md);
	}

	.module-list li {
		padding: var(--space-sm) 0;
		border-bottom: 1px solid var(--border);
		color: var(--text-muted);
	}

	.module-list code {
		color: var(--accent);
		font-weight: 500;
	}
</style>
