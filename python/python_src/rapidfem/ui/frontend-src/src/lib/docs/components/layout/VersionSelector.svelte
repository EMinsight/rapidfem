<script lang="ts">
	import { base } from '$app/paths';
	import { goto } from '$app/navigation';
	import Icon from '$lib/docs/components/common/Icon.svelte';
	import type { VersionManifest } from '$lib/docs/api/types';

	interface Props {
		manifest: VersionManifest;
		currentVersion: string;
	}

	let { manifest, currentVersion }: Props = $props();

	let open = $state(false);

	// Label shown for the active version — 'latest' resolves to its tag.
	let currentLabel = $derived(
		currentVersion === 'latest' ? `${manifest.latest} (latest)` : currentVersion
	);

	function select(version: string) {
		open = false;
		if (version !== currentVersion) {
			goto(`${base}/${version}/api/`);
		}
	}

	function handleBlur(e: FocusEvent) {
		const next = e.relatedTarget as Node | null;
		if (!next || !(e.currentTarget as HTMLElement).contains(next)) {
			open = false;
		}
	}
</script>

<div class="version-selector" onfocusout={handleBlur}>
	<button class="version-trigger" onclick={() => (open = !open)}>
		<span class="version-label">{currentLabel}</span>
		<Icon name={open ? 'chevron-up' : 'chevron-down'} size={12} />
	</button>

	{#if open}
		<div class="version-menu">
			<button
				class="version-option"
				class:active={currentVersion === 'latest'}
				onclick={() => select('latest')}
			>
				latest
				<span class="version-tag">{manifest.latest}</span>
			</button>
			{#each manifest.versions as v}
				<button
					class="version-option"
					class:active={currentVersion === v.tag}
					onclick={() => select(v.tag)}
				>
					{v.tag}
					<span class="version-tag">{v.date}</span>
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.version-selector {
		position: relative;
		padding: var(--space-md);
		padding-bottom: 0;
	}

	.version-trigger {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-sm);
		width: 100%;
		padding: var(--space-sm) var(--space-md);
		background: var(--surface-inset);
		border: 1px solid var(--border);
		color: var(--text);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 500;
		text-transform: none;
		letter-spacing: normal;
		cursor: pointer;
	}

	.version-trigger:hover {
		border-color: var(--border-focus);
		background: var(--surface-inset);
	}

	.version-label {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.version-menu {
		position: absolute;
		top: calc(100% + 2px);
		left: var(--space-md);
		right: var(--space-md);
		z-index: var(--z-dropdown);
		background: var(--surface-raised);
		border: 1px solid var(--border);
		box-shadow: var(--shadow-md);
		max-height: 280px;
		overflow-y: auto;
	}

	.version-option {
		display: flex;
		align-items: baseline;
		justify-content: space-between;
		gap: var(--space-sm);
		width: 100%;
		padding: var(--space-sm) var(--space-md);
		background: none;
		border: none;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		text-transform: none;
		letter-spacing: normal;
		color: var(--text-muted);
		text-align: left;
		cursor: pointer;
		transition: all var(--transition-fast);
	}

	.version-option:hover {
		background: var(--surface-hover);
		color: var(--text);
	}

	.version-option.active {
		color: var(--accent);
	}

	.version-tag {
		font-size: var(--fs-xs);
		color: var(--text-disabled);
	}
</style>
