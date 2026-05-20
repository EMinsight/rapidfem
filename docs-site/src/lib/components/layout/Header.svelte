<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/stores';
	import { external } from '$lib/config/rapidfem';

	// Top navigation bar — markup and styling taken from the RapidFEM demo
	// header so the docs sit visually flush with the rest of the site.
	let path = $derived($page.url.pathname.replace(/\/$/, ''));
	let isApi = $derived(path.includes('/api'));
	let isOverview = $derived(!isApi);
</script>

<header>
	<a class="brand" href="{base}/" aria-label="RapidFEM">
		<img src="{base}/favicon.svg" alt="RapidFEM" class="logo" />
	</a>
	<span class="nav-sep"></span>
	<nav class="tabs">
		<a class="tab" class:active={isOverview} href="{base}/">Overview</a>
		<a class="tab" class:active={isApi} href="{base}/latest/api/">API</a>
	</nav>
	<nav class="tabs right">
		<a class="tab" href={external.demo}>Examples</a>
		<a class="tab" href={external.github}>GitHub</a>
	</nav>
</header>

<style>
	header {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		height: var(--header-height);
		padding: 0 var(--space-lg);
		background: var(--surface-raised);
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
		gap: 0;
		height: 100%;
	}

	.tabs.right {
		margin-left: auto;
	}

	.tab {
		display: flex;
		align-items: center;
		padding: 0 14px;
		font-size: var(--font-xs);
		font-weight: 600;
		font-family: var(--font-mono);
		letter-spacing: 0.5px;
		color: var(--text-disabled);
		text-decoration: none;
		transition: color var(--transition-fast);
	}

	.tab:hover {
		color: var(--text-muted);
		text-decoration: none;
	}

	.tab.active {
		color: var(--accent);
	}
</style>
