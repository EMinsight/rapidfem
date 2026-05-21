<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import type { EditorView } from '@codemirror/view';
	import { createViewer } from '$lib/docs/codemirror';

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

	// Keep the editor document in sync when `code` changes — a single
	// CodeBlock instance is reused across prop changes (e.g. the Quick
	// Start FD/TD tab switch), so the view must be patched, not rebuilt.
	$effect(() => {
		if (view && code !== view.state.doc.toString()) {
			view.dispatch({
				changes: { from: 0, to: view.state.doc.length, insert: code }
			});
		}
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
