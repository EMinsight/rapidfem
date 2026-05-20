<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { page } from '$app/stores';
	import Icon from '$lib/components/common/Icon.svelte';
	import VersionSelector from './VersionSelector.svelte';
	import { ApiToc } from '$lib/components/api';
	import { getSidebarItems, external } from '$lib/config/rapidfem';
	import type { APIPackage, VersionManifest } from '$lib/api/types';
	import { buildSearchItems, createSearch, runSearch, type SearchItem } from '$lib/search';
	import { searchTarget } from '$lib/stores/searchNavigation';

	interface Props {
		api: APIPackage;
		manifest: VersionManifest;
		currentVersion: string;
	}

	let { api, manifest, currentVersion }: Props = $props();

	let query = $state('');
	let searchInput = $state<HTMLInputElement | null>(null);

	// Sorted module list drives the table of contents.
	let modules = $derived(Object.values(api.modules));

	let fuse = $derived(createSearch(buildSearchItems(api)));
	let results = $derived(runSearch(fuse, query, 25));
	let showResults = $derived(query.trim().length >= 2);

	let navItems = $derived(getSidebarItems(currentVersion));

	function isActive(path: string): boolean {
		return $page.url.pathname.replace(/\/$/, '') === `${base}/${path}`.replace(/\/$/, '');
	}

	function handleGlobalKeydown(event: KeyboardEvent) {
		if ((event.ctrlKey || event.metaKey) && event.key === 'f') {
			event.preventDefault();
			searchInput?.focus();
		}
	}

	onMount(() => {
		window.addEventListener('keydown', handleGlobalKeydown);
		return () => window.removeEventListener('keydown', handleGlobalKeydown);
	});

	function handleSearchKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			if (query) query = '';
			else searchInput?.blur();
			event.stopPropagation();
		}
	}

	function selectResult(item: SearchItem) {
		searchTarget.set({
			name: item.name,
			type: item.type,
			parentClass: item.parentClass,
			source: 'search'
		});
		query = '';
	}
</script>

<aside class="sidebar">
	<div class="search-container">
		<Icon name="search" size={14} />
		<input
			type="text"
			placeholder="Search the API…"
			bind:value={query}
			bind:this={searchInput}
			onkeydown={handleSearchKeydown}
		/>
		{#if query}
			<button class="icon-btn clear-btn" onclick={() => (query = '')} aria-label="Clear">
				<Icon name="x" size={12} />
			</button>
		{/if}
	</div>

	{#if showResults}
		<nav class="search-results">
			{#if results.length > 0}
				{#each results as result}
					<button class="search-result" onclick={() => selectResult(result)}>
						<span class="result-name">{result.name}</span>
						<span class="result-meta">
							<span class="result-type">{result.type}</span>
							<span class="result-module">{result.module}</span>
						</span>
					</button>
				{/each}
			{:else}
				<div class="no-results">No results for "{query}"</div>
			{/if}
		</nav>
	{:else}
		<div class="sidebar-fixed">
			<VersionSelector {manifest} {currentVersion} />
			<nav class="sidebar-nav">
				{#each navItems as item}
					{#if item.external}
						<a href={item.path} class="sidebar-item">
							{#if item.icon}<Icon name={item.icon} size={14} />{/if}
							<span>{item.title}</span>
							<Icon name="external-link" size={12} />
						</a>
					{:else}
						<a
							href="{base}/{item.path}"
							class="sidebar-item"
							class:active={isActive(item.path)}
						>
							{#if item.icon}<Icon name={item.icon} size={14} />{/if}
							<span>{item.title}</span>
						</a>
					{/if}
				{/each}
			</nav>
		</div>
		<div class="sidebar-scrollable">
			<ApiToc {modules} />
		</div>
	{/if}
</aside>

<style>
	.sidebar {
		width: var(--sidebar-width);
		flex-shrink: 0;
		display: flex;
		flex-direction: column;
		background: var(--surface);
		border-right: 1px solid var(--border);
	}

	.search-container {
		flex-shrink: 0;
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		height: var(--header-height);
		padding: 0 var(--space-md);
		border-bottom: 1px solid var(--border);
		color: var(--text-muted);
	}

	.search-container:focus-within {
		color: var(--accent);
	}

	.search-container input {
		flex: 1;
		min-width: 0;
		background: none;
		border: none;
		padding: 0;
		font-family: var(--font-ui);
		color: var(--text);
	}

	.search-container input:focus {
		outline: none;
	}

	.clear-btn {
		width: 20px;
		height: 20px;
		flex-shrink: 0;
	}

	.sidebar-fixed {
		flex-shrink: 0;
	}

	.sidebar-scrollable {
		flex: 1;
		overflow-y: auto;
		min-height: 0;
		border-top: 1px solid var(--border);
	}

	.search-results {
		flex: 1;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
	}

	.search-result {
		display: flex;
		flex-direction: column;
		gap: 2px;
		padding: var(--space-sm) var(--space-md);
		background: none;
		border: none;
		border-bottom: 1px solid var(--border-subtle);
		text-transform: none;
		letter-spacing: normal;
		text-align: left;
		cursor: pointer;
		transition: background var(--transition-fast);
	}

	.search-result:hover {
		background: var(--surface-hover);
	}

	.result-name {
		font-family: var(--font-mono);
		font-size: var(--font-base);
		font-weight: 600;
		color: var(--accent);
	}

	.result-meta {
		display: flex;
		gap: var(--space-sm);
		font-size: var(--font-base);
	}

	.result-type {
		text-transform: uppercase;
		letter-spacing: 0.5px;
		color: var(--text-disabled);
		font-size: 10px;
		align-self: center;
	}

	.result-module {
		font-family: var(--font-mono);
		color: var(--text-muted);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.no-results {
		padding: var(--space-lg);
		text-align: center;
		color: var(--text-muted);
	}

	.sidebar-nav {
		display: flex;
		flex-direction: column;
		gap: var(--space-xs);
		padding: var(--space-md);
	}

	.sidebar-item {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		padding: var(--space-sm) var(--space-md);
		font-size: var(--font-base);
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		text-decoration: none;
		transition: all var(--transition-fast);
	}

	.sidebar-item:hover {
		color: var(--text);
		background: var(--surface-hover);
		text-decoration: none;
	}

	.sidebar-item.active {
		color: var(--accent);
		background: var(--accent-bg);
	}

	.sidebar-item span {
		flex: 1;
	}
</style>
