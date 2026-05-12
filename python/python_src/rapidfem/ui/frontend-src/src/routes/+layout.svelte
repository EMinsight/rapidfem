<script lang="ts">
	import '../app.css';
	import Modal from '$lib/components/Modal.svelte';
	let { children } = $props();
</script>

{@render children()}
<Modal />
