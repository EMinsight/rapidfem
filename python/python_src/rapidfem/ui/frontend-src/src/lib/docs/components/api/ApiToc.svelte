<script lang="ts">
	import { onDestroy } from 'svelte';
	import type { APIModule } from '$lib/docs/api/types';
	import Icon from '$lib/docs/components/common/Icon.svelte';
	import { searchTarget } from '$lib/docs/stores/searchNavigation';

	interface Props {
		modules: APIModule[];
	}

	interface TreeNode {
		name: string;
		fullPath: string;
		module: APIModule | null;
		children: Map<string, TreeNode>;
	}

	let { modules }: Props = $props();

	let navigationTimeout: ReturnType<typeof setTimeout> | null = null;

	function setNavigationTimeout() {
		if (navigationTimeout) clearTimeout(navigationTimeout);
		navigationTimeout = setTimeout(() => {
			isNavigating = false;
			navigationTimeout = null;
		}, 60);
	}

	onDestroy(() => {
		if (navigationTimeout) clearTimeout(navigationTimeout);
	});

	let moduleTree = $derived.by(() => {
		const root: TreeNode = { name: '', fullPath: '', module: null, children: new Map() };
		for (const mod of modules) {
			const parts = mod.name.split('.');
			let current = root;
			for (let i = 0; i < parts.length; i++) {
				const part = parts[i];
				const fullPath = parts.slice(0, i + 1).join('.');
				if (!current.children.has(part)) {
					current.children.set(part, {
						name: part,
						fullPath,
						module: null,
						children: new Map()
					});
				}
				current = current.children.get(part)!;
			}
			current.module = mod;
		}
		return root;
	});

	let expandedGroups = $state<Set<string>>(new Set());
	let activeId = $state<string | null>(null);
	let isNavigating = $state(false);

	function toggleGroup(path: string) {
		const next = new Set(expandedGroups);
		if (next.has(path)) next.delete(path);
		else next.add(path);
		expandedGroups = next;
	}

	function getModuleId(moduleName: string): string {
		return moduleName.replace(/\./g, '-');
	}

	function scrollToElement(id: string) {
		const element = document.getElementById(id);
		if (element) {
			element.scrollIntoView({ block: 'start' });
			activeId = id;
		}
	}

	function navigateToClass(className: string) {
		isNavigating = true;
		activeId = className;
		searchTarget.set({ name: className, type: 'class', source: 'toc' });
		setNavigationTimeout();
	}

	function navigateToFunction(funcName: string) {
		isNavigating = true;
		activeId = funcName;
		searchTarget.set({ name: funcName, type: 'function', source: 'toc' });
		setNavigationTimeout();
	}

	function getSortedChildren(node: TreeNode): [string, TreeNode][] {
		return Array.from(node.children.entries()).sort((a, b) => a[0].localeCompare(b[0]));
	}

	function hasContent(node: TreeNode): boolean {
		if (node.module && (node.module.classes.length > 0 || node.module.functions.length > 0)) {
			return true;
		}
		for (const child of node.children.values()) {
			if (hasContent(child)) return true;
		}
		return false;
	}

	// Expand every module group initially so the whole API is visible.
	$effect(() => {
		const all = new Set<string>();
		for (const mod of modules) {
			const parts = mod.name.split('.');
			for (let i = 1; i <= parts.length; i++) {
				all.add(parts.slice(0, i).join('.'));
			}
		}
		expandedGroups = all;
	});

	// Track the section in view to highlight the matching TOC entry.
	$effect(() => {
		if (typeof window === 'undefined') return;
		const scrollContainer = document.querySelector('.doc-main');

		const observer = new IntersectionObserver(
			(entries) => {
				if (isNavigating) return;
				for (const entry of entries) {
					if (entry.isIntersecting) activeId = entry.target.id;
				}
			},
			{ root: scrollContainer, rootMargin: '-10% 0px -80% 0px', threshold: 0 }
		);

		for (const mod of modules) {
			const moduleEl = document.getElementById(getModuleId(mod.name));
			if (moduleEl) observer.observe(moduleEl);
			for (const cls of mod.classes) {
				const classEl = document.getElementById(cls.name);
				if (classEl) observer.observe(classEl);
			}
		}

		return () => observer.disconnect();
	});
</script>

{#snippet treeItem(node: TreeNode, depth: number)}
	{@const hasChildren = node.children.size > 0}
	{@const isExpanded = expandedGroups.has(node.fullPath)}
	{@const nodeId = getModuleId(node.fullPath)}
	{@const hasClasses = !!node.module?.classes.length}
	{@const hasFunctions = !!node.module?.functions.length}
	{@const hasModuleContent = hasClasses || hasFunctions}

	{#if hasContent(node)}
		<div class="toc-item" style="--depth: {depth}">
			<button
				class="toc-node"
				class:active={activeId === nodeId}
				class:has-children={hasChildren || hasModuleContent}
				onclick={() => {
					if (node.module) scrollToElement(nodeId);
					if (hasChildren || hasModuleContent) toggleGroup(node.fullPath);
				}}
			>
				{#if hasChildren || hasModuleContent}
					<span class="toc-icon" class:expanded={isExpanded}>
						<Icon name="chevron-down" size={12} />
					</span>
				{/if}
				<span class="toc-name">{node.name}</span>
			</button>

			{#if isExpanded}
				<div class="toc-children">
					{#each getSortedChildren(node) as [, child]}
						{@render treeItem(child, depth + 1)}
					{/each}
					{#if hasClasses}
						{#each node.module!.classes as cls}
							<button
								class="toc-leaf"
								class:active={activeId === cls.name}
								style="--depth: {depth + 1}"
								onclick={() => navigateToClass(cls.name)}
							>
								{cls.name}
							</button>
						{/each}
					{/if}
					{#if hasFunctions}
						{#each node.module!.functions as func}
							<button
								class="toc-leaf"
								class:active={activeId === func.name}
								style="--depth: {depth + 1}"
								onclick={() => navigateToFunction(func.name)}
							>
								{func.name}()
							</button>
						{/each}
					{/if}
				</div>
			{/if}
		</div>
	{/if}
{/snippet}

<div class="api-toc">
	<nav class="api-toc-nav">
		{#each getSortedChildren(moduleTree) as [, rootNode]}
			{@render treeItem(rootNode, 0)}
		{/each}
	</nav>
</div>

<style>
	/* Tree styled to match the RapidFEM notebook file browser. */
	.api-toc {
		display: flex;
		flex-direction: column;
		padding: var(--space-sm) 0 var(--space-md);
	}

	.api-toc-nav {
		display: flex;
		flex-direction: column;
	}

	.toc-item {
		display: flex;
		flex-direction: column;
	}

	/* Module group row — like a folder row */
	.toc-node {
		display: flex;
		align-items: center;
		justify-content: flex-start;
		gap: 4px;
		width: 100%;
		padding: 3px var(--space-lg);
		padding-left: calc(var(--space-lg) + var(--depth, 0) * 12px);
		background: none;
		border: none;
		border-left: 2px solid transparent;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		font-weight: 400;
		text-transform: none;
		letter-spacing: 0;
		color: var(--text-muted);
		text-align: left;
		cursor: pointer;
		transition: background var(--transition-fast), color var(--transition-fast);
	}

	/* Class / function leaf — like a file row */
	.toc-leaf {
		display: flex;
		justify-content: flex-start;
		width: 100%;
		padding: 3px var(--space-lg);
		padding-left: calc(var(--space-lg) + 16px + var(--depth, 0) * 12px);
		background: none;
		border: none;
		border-left: 2px solid transparent;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		font-weight: 400;
		text-transform: none;
		letter-spacing: 0;
		color: var(--text-muted);
		text-align: left;
		cursor: pointer;
		transition: background var(--transition-fast), color var(--transition-fast);
	}

	.toc-node:hover,
	.toc-leaf:hover {
		color: var(--text);
		background: var(--surface-panel);
	}

	.toc-node.active,
	.toc-leaf.active {
		color: var(--accent);
		background: var(--accent-bg);
		border-left-color: var(--accent);
	}

	.toc-node:not(.has-children) {
		padding-left: calc(var(--space-lg) + 16px + var(--depth, 0) * 12px);
	}

	.toc-icon {
		display: flex;
		align-items: center;
		justify-content: center;
		width: 12px;
		flex-shrink: 0;
		color: var(--text-dim);
		transform: rotate(-90deg);
		transition: transform var(--transition-fast);
	}

	.toc-icon.expanded {
		transform: rotate(0deg);
	}

	.toc-name {
		font-family: var(--font-mono);
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.toc-children {
		display: flex;
		flex-direction: column;
	}
</style>
