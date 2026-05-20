<script lang="ts">
	import { tick } from 'svelte';
	import type { APIClass } from '$lib/docs/api/types';
	import FunctionDoc from './FunctionDoc.svelte';
	import DocstringRenderer from './DocstringRenderer.svelte';
	import TypeRef from './TypeRef.svelte';
	import CodeMirror from '$lib/docs/components/common/CodeMirror.svelte';
	import Icon from '$lib/docs/components/common/Icon.svelte';
	import { tooltip } from '$lib/docs/components/common/Tooltip.svelte';
	import { searchTarget, clearSearchTarget } from '$lib/docs/stores/searchNavigation';

	interface Props {
		cls: APIClass;
		expanded?: boolean;
	}

	let { cls, expanded = false }: Props = $props();

	let isExpanded = $state((() => expanded)());
	let viewMode = $state<'docs' | 'source'>('docs');
	let tileElement: HTMLDivElement | undefined = $state();

	// __init__ parameters are surfaced through the class docstring.
	let methods = $derived(cls.methods.filter((m) => m.name !== '__init__'));

	function toggleView(e: MouseEvent | KeyboardEvent) {
		e.stopPropagation();
		viewMode = viewMode === 'docs' ? 'source' : 'docs';
	}

	$effect(() => {
		const target = $searchTarget;
		if (!target) return;

		if (target.type === 'class' && target.name === cls.name) {
			isExpanded = true;
			clearSearchTarget();
			tick().then(() => tileElement?.scrollIntoView({ block: 'start' }));
		} else if (target.type === 'method' && target.parentClass === cls.name) {
			isExpanded = true;
			// Leave the target set — FunctionDoc handles the final scroll.
		}
	});
</script>

<div class="tile class-tile" id={cls.name} bind:this={tileElement}>
	<button
		class="panel-header class-header"
		class:expanded={isExpanded}
		onclick={() => (isExpanded = !isExpanded)}
	>
		<div class="class-header-content">
			<div class="class-header-top">
				<code class="class-name">{cls.name}</code>
				{#if cls.bases && cls.bases.length > 0}
					<span class="class-bases"
						>({#each cls.bases as base, i}{#if i > 0}, {/if}<TypeRef
								type={base}
							/>{/each})</span
					>
				{/if}
			</div>
			{#if cls.description}
				<span class="class-desc">{cls.description}</span>
			{/if}
		</div>
		<div class="header-actions">
			{#if cls.source && isExpanded}
				<span
					role="button"
					tabindex="0"
					class="icon-btn"
					onclick={toggleView}
					onkeydown={(e) => e.key === 'Enter' && toggleView(e)}
					use:tooltip={viewMode === 'docs' ? 'View source' : 'View docs'}
				>
					<Icon name={viewMode === 'docs' ? 'braces' : 'book'} size={14} />
				</span>
			{/if}
			<span class="icon-btn chevron">
				<Icon name={isExpanded ? 'chevron-up' : 'chevron-down'} size={14} />
			</span>
		</div>
	</button>

	{#if isExpanded}
		{#if viewMode === 'docs'}
			<div class="panel-body class-body">
				{#if cls.docstring_html}
					<DocstringRenderer html={cls.docstring_html} />
				{/if}

				{#if methods.length > 0}
					<div class="methods-section">
						<div class="label-uppercase methods-header">Methods</div>
						<div class="methods-list">
							{#each methods as method}
								<FunctionDoc func={method} isMethod={true} parentClass={cls.name} />
							{/each}
						</div>
					</div>
				{/if}
			</div>
		{:else}
			<div class="source-body">
				<CodeMirror code={cls.source ?? ''} />
			</div>
		{/if}
	{/if}
</div>

<style>
	.class-tile {
		margin-bottom: var(--space-lg);
	}

	.class-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-sm);
		width: 100%;
		border-bottom: none;
		text-transform: none;
		letter-spacing: normal;
		text-align: left;
		cursor: pointer;
	}

	.class-header:hover {
		background: var(--surface-raised);
	}

	.class-header.expanded {
		border-bottom: 1px solid var(--border);
	}

	.class-header-content {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: var(--space-xs);
		flex: 1;
		min-width: 0;
	}

	.class-header-top {
		display: flex;
		align-items: baseline;
		gap: var(--space-xs);
		flex-wrap: wrap;
	}

	.class-name {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		font-weight: 600;
		color: var(--accent);
		background: none;
		border: none;
		padding: 0;
	}

	.class-bases {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
	}

	.class-desc {
		font-size: var(--fs-xs);
		color: var(--text-muted);
		line-height: 1.5;
		text-transform: none;
		font-weight: 400;
	}

	.header-actions {
		display: flex;
		align-items: center;
		gap: var(--space-xs);
		flex-shrink: 0;
	}

	.chevron {
		pointer-events: none;
	}

	.source-body {
		border-top: 1px solid var(--border);
	}

	.methods-section {
		position: relative;
		margin-top: var(--space-xl);
		padding-top: var(--space-lg);
	}

	.methods-section::before {
		content: '';
		position: absolute;
		top: 0;
		left: calc(-1 * var(--space-lg));
		right: calc(-1 * var(--space-lg));
		height: 1px;
		background: var(--border);
	}

	.methods-header {
		margin-bottom: var(--space-md);
	}

	.methods-list {
		display: flex;
		flex-direction: column;
		gap: var(--space-md);
	}
</style>
