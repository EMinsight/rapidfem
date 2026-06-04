<script lang="ts">
	import { onMount } from 'svelte';
	import '../app.css';
	import Modal from '$lib/components/Modal.svelte';
	import { initTooltips } from '$lib/tooltip';
	let { children } = $props();

	onMount(() => initTooltips());
</script>

{@render children()}
<Modal />
