<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { EditorState, type Extension } from '@codemirror/state';
	import { EditorView, keymap, lineNumbers, highlightActiveLine, drawSelection, dropCursor } from '@codemirror/view';
	import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
	import { python } from '@codemirror/lang-python';
	import { bracketMatching, indentOnInput, syntaxHighlighting, defaultHighlightStyle } from '@codemirror/language';

	let {
		value = $bindable<string>(''),
		onSave,
	}: {
		value: string;
		onSave?: (text: string) => void;
	} = $props();

	let host: HTMLDivElement | undefined = $state();
	let view: EditorView | null = null;
	let last_set_value = '';

	function build(initial: string): EditorView {
		const save_keymap = keymap.of([
			{
				key: 'Mod-s',
				preventDefault: true,
				run: (v) => {
					onSave?.(v.state.doc.toString());
					return true;
				},
			},
		]);
		const extensions: Extension[] = [
			lineNumbers(),
			history(),
			drawSelection(),
			dropCursor(),
			indentOnInput(),
			bracketMatching(),
			highlightActiveLine(),
			syntaxHighlighting(defaultHighlightStyle),
			python(),
			save_keymap,
			keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
			EditorView.updateListener.of((upd) => {
				if (upd.docChanged) {
					const text = upd.state.doc.toString();
					last_set_value = text;
					value = text;
				}
			}),
			EditorView.theme({
				'&': { height: '100%', fontSize: '13px' },
				'.cm-scroller': { fontFamily: 'ui-monospace, Consolas, Menlo, monospace' },
				'&.cm-focused': { outline: 'none' },
			}),
		];
		const state = EditorState.create({ doc: initial, extensions });
		return new EditorView({ state, parent: host! });
	}

	onMount(() => {
		if (!host) return;
		view = build(value);
		last_set_value = value;
	});

	onDestroy(() => view?.destroy());

	// Sync external changes to the editor without echoing them back as
	// internal edits (the updateListener would otherwise loop).
	$effect(() => {
		if (!view) return;
		if (value === last_set_value) return;
		const current = view.state.doc.toString();
		if (current === value) {
			last_set_value = value;
			return;
		}
		view.dispatch({
			changes: { from: 0, to: current.length, insert: value },
		});
		last_set_value = value;
	});
</script>

<div bind:this={host} class="editor"></div>

<style>
	.editor {
		height: 100%;
		width: 100%;
		background: #0e0e0e;
		color: #e8e8e8;
		overflow: hidden;
	}
	:global(.cm-editor) { height: 100%; }
	:global(.cm-content) { caret-color: #e8e8e8; }
	:global(.cm-gutters) { background: #141414; border-right: 1px solid #222; color: #555; }
	:global(.cm-activeLine) { background: rgba(255,255,255,0.03); }
	:global(.cm-activeLineGutter) { background: rgba(255,255,255,0.05); color: #aaa; }
	:global(.cm-selectionBackground), :global(.cm-content ::selection) { background: rgba(90, 170, 130, 0.25) !important; }
</style>
