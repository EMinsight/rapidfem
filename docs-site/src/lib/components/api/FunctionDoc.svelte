<script lang="ts">
	import { tick } from 'svelte';
	import type { APIFunction, APIMethod } from '$lib/api/types';
	import DocstringRenderer from './DocstringRenderer.svelte';
	import TypeRef from './TypeRef.svelte';
	import CodeMirror from '$lib/components/common/CodeMirror.svelte';
	import Icon from '$lib/components/common/Icon.svelte';
	import { tooltip } from '$lib/components/common/Tooltip.svelte';
	import { searchTarget, clearSearchTarget } from '$lib/stores/searchNavigation';

	interface Props {
		func: APIFunction | APIMethod;
		isMethod?: boolean;
		expanded?: boolean;
		parentClass?: string;
	}

	let { func, isMethod = false, expanded = false, parentClass }: Props = $props();

	let elementId = $derived(parentClass ? `${parentClass}.${func.name}` : func.name);
	let isExpanded = $state((() => expanded)());
	let viewMode = $state<'docs' | 'source'>('docs');
	let tileElement: HTMLDivElement | undefined = $state();

	let methodType = $derived((func as APIMethod).method_type);
	let showBadge = $derived(isMethod && methodType && methodType !== 'method');

	function toggleView(e: MouseEvent | KeyboardEvent) {
		e.stopPropagation();
		viewMode = viewMode === 'docs' ? 'source' : 'docs';
	}

	$effect(() => {
		const target = $searchTarget;
		if (!target) return;

		const nameMatches = target.name === func.name;
		const isMethodMatch =
			target.type === 'method' && nameMatches && target.parentClass === parentClass;
		const isFunctionMatch = target.type === 'function' && nameMatches && !parentClass;

		if (isMethodMatch || isFunctionMatch) {
			isExpanded = true;
			clearSearchTarget();
			tick().then(() => tileElement?.scrollIntoView({ block: 'start' }));
		}
	});
</script>

<div class="tile method-tile" id={elementId} bind:this={tileElement}>
	<button
		class="panel-header method-header"
		class:expanded={isExpanded}
		onclick={() => (isExpanded = !isExpanded)}
	>
		<div class="method-header-content">
			<code class="method-name">{func.name}</code>
			{#if func.signature}
				<code class="method-signature">{func.signature}</code>
			{/if}
			{#if showBadge}
				<span class="badge accent">{methodType}</span>
			{/if}
		</div>
		<div class="header-actions">
			{#if func.source && isExpanded}
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
			<div class="panel-body method-body">
				{#if func.docstring_html}
					<DocstringRenderer html={func.docstring_html} />
				{:else if func.description}
					<p class="method-desc">{func.description}</p>
				{:else}
					<p class="method-desc empty">No description available.</p>
				{/if}

				{#if func.returns}
					<div class="method-returns">
						<span class="label-uppercase">Returns</span>
						<TypeRef type={func.returns} />
					</div>
				{/if}
			</div>
		{:else}
			<div class="source-body">
				<CodeMirror code={func.source ?? ''} />
			</div>
		{/if}
	{/if}
</div>

<style>
	.method-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-sm);
		padding: var(--space-sm) var(--space-md);
		text-transform: none;
		letter-spacing: normal;
		width: 100%;
		border-bottom: none;
		text-align: left;
		cursor: pointer;
	}

	.method-header:hover {
		background: var(--surface-raised);
	}

	.method-header.expanded {
		border-bottom: 1px solid var(--border);
	}

	.method-header-content {
		display: flex;
		align-items: center;
		gap: var(--space-xs);
		flex-wrap: wrap;
		flex: 1;
		min-width: 0;
	}

	.method-name {
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		font-weight: 600;
		color: var(--accent);
		background: none;
		border: none;
		padding: 0;
		flex-shrink: 0;
	}

	.method-signature {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
		background: none;
		border: none;
		padding: 0;
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

	.method-body {
		padding: var(--space-md);
	}

	.method-desc {
		font-size: var(--fs-xs);
		color: var(--text-muted);
		margin: 0;
		line-height: 1.6;
	}

	.method-desc.empty {
		font-style: italic;
		color: var(--text-disabled);
	}

	.source-body {
		border-top: 1px solid var(--border);
	}

	.method-returns {
		display: flex;
		align-items: baseline;
		gap: var(--space-sm);
		margin-top: var(--space-md);
		padding-top: var(--space-md);
		border-top: 1px solid var(--border);
	}
</style>
