<script lang="ts">
	import { tick } from 'svelte';
	import type { APIModule } from '$lib/docs/api/types';
	import ClassDoc from './ClassDoc.svelte';
	import FunctionDoc from './FunctionDoc.svelte';
	import DocstringRenderer from './DocstringRenderer.svelte';
	import { searchTarget, clearSearchTarget } from '$lib/docs/stores/searchNavigation';

	interface Props {
		module: APIModule;
	}

	let { module }: Props = $props();
	let sectionElement: HTMLElement | undefined = $state();

	$effect(() => {
		const target = $searchTarget;
		if (!target) return;
		if (target.type === 'module' && target.name === module.name) {
			clearSearchTarget();
			tick().then(() => sectionElement?.scrollIntoView({ block: 'start' }));
		}
	});
</script>

<section class="api-module" id={module.name.replace(/\./g, '-')} bind:this={sectionElement}>
	<header class="api-module-header">
		<h3 class="api-module-name"><code>{module.name}</code></h3>
		{#if module.description}
			<p class="api-module-desc">{module.description}</p>
		{/if}
	</header>

	{#if module.docstring_html}
		<div class="api-module-docstring">
			<DocstringRenderer html={module.docstring_html} />
		</div>
	{/if}

	{#if module.classes.length > 0}
		<div class="api-module-classes">
			{#each module.classes as cls}
				<ClassDoc {cls} />
			{/each}
		</div>
	{/if}

	{#if module.functions.length > 0}
		<div class="api-module-functions">
			{#each module.functions as func}
				<FunctionDoc {func} />
			{/each}
		</div>
	{/if}
</section>

<style>
	.api-module {
		position: relative;
		margin-bottom: var(--space-3xl);
		padding-top: var(--space-2xl);
		scroll-margin-top: var(--space-lg);
	}

	.api-module::before {
		content: '';
		position: absolute;
		top: 0;
		left: -50vw;
		width: 200vw;
		height: 1px;
		background: var(--border);
	}

	.api-module:first-child {
		padding-top: 0;
	}

	.api-module:first-child::before {
		display: none;
	}

	.api-module-header {
		margin-bottom: var(--space-lg);
	}

	.api-module-name {
		margin: 0 0 var(--space-xs);
	}

	.api-module-name code {
		font-family: var(--font-mono);
		font-size: var(--fs-md);
		font-weight: 600;
		color: var(--accent);
		background: none;
		border: none;
		padding: 0;
	}

	.api-module-desc {
		font-size: var(--fs-xs);
		color: var(--text-muted);
		margin: 0;
		line-height: 1.6;
	}

	.api-module-docstring {
		margin-top: var(--space-lg);
		margin-bottom: var(--space-xl);
	}

	.api-module-classes {
		display: flex;
		flex-direction: column;
		gap: var(--space-lg);
	}

	.api-module-functions {
		display: flex;
		flex-direction: column;
		gap: var(--space-md);
		margin-top: var(--space-lg);
	}
</style>
