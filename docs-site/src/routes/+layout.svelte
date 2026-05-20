<script lang="ts">
	import '../app.css';
	import { afterNavigate } from '$app/navigation';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import { Header } from '$lib/components/layout';

	let { children } = $props();

	// Reset scroll position on navigation.
	afterNavigate(() => {
		document.querySelectorAll('.main-content, .doc-main').forEach((el) => {
			if (el instanceof HTMLElement) el.scrollTop = 0;
		});
		window.scrollTo(0, 0);
	});
</script>

<Tooltip />

<a href="#main-content" class="skip-link">Skip to main content</a>

<div class="app">
	<Header />
	<div id="main-content" class="main-content">
		{@render children()}
	</div>
</div>

<style>
	.app {
		height: 100vh;
		min-width: var(--app-min-width);
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.main-content {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow-y: auto;
		min-height: 0;
		min-width: 0;
	}
</style>
