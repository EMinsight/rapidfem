<script lang="ts">
	import type { Snippet } from 'svelte';
	import Sidebar from './Sidebar.svelte';
	import type { APIPackage, VersionManifest } from '$lib/docs/api/types';

	interface Props {
		api: APIPackage;
		manifest: VersionManifest;
		currentVersion: string;
		children: Snippet;
	}

	let { api, manifest, currentVersion, children }: Props = $props();
</script>

<div class="doc-layout">
	<Sidebar {api} {manifest} {currentVersion} />
	<div class="doc-main">
		<div class="doc-content">
			{@render children()}
		</div>
	</div>
</div>

<style>
	.doc-layout {
		display: flex;
		flex: 1;
		min-height: 0;
		min-width: 0;
		overflow: hidden;
	}

	@media (max-width: 768px) {
		.doc-layout :global(.sidebar) {
			display: none;
		}
	}

	.doc-main {
		flex: 1;
		display: flex;
		min-width: 0;
		min-height: 0;
		background: var(--surface);
		overflow-x: hidden;
		overflow-y: auto;
	}

	.doc-content {
		flex: 1;
		min-width: 0;
		max-width: var(--content-max-width);
		margin: 0 auto;
		padding: var(--space-xl);
		padding-bottom: var(--space-4xl);
	}

	@media (max-width: 600px) {
		.doc-content {
			padding: var(--space-md);
			padding-bottom: var(--space-2xl);
		}
	}
</style>
