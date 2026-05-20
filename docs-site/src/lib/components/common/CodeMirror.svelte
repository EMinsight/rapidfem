<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import type { EditorView } from '@codemirror/view';
	import { createViewer } from '$lib/codemirror';

	interface Props {
		code: string;
		lineNumbers?: boolean;
	}

	let { code, lineNumbers = true }: Props = $props();

	let host: HTMLDivElement | undefined = $state();
	let view: EditorView | null = null;

	onMount(() => {
		if (host) view = createViewer(host, code, { lineNumbers });
	});

	onDestroy(() => view?.destroy());
</script>

<div class="cm" bind:this={host}></div>

<style>
	.cm {
		background: var(--surface-mid);
		overflow: hidden;
	}

	.cm :global(.cm-editor) {
		max-height: 520px;
	}
</style>
