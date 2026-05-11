<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { EditorState, type Extension } from '@codemirror/state';
	import { EditorView, keymap, lineNumbers, highlightActiveLine, highlightActiveLineGutter, drawSelection, dropCursor } from '@codemirror/view';
	import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
	import { python } from '@codemirror/lang-python';
	import { bracketMatching, indentOnInput, syntaxHighlighting, HighlightStyle, indentUnit } from '@codemirror/language';
	import { tags as t } from '@lezer/highlight';
	import { palette, fonts } from '$lib/theme';

	let {
		index,
		source = $bindable<string>(''),
		status = 'idle' as 'idle' | 'running' | 'ok' | 'error',
		onRun,
		onRunAllBelow,
		onFocus,
		focused = false,
	}: {
		index: number;
		source: string;
		status?: 'idle' | 'running' | 'ok' | 'error';
		onRun: () => void;
		onRunAllBelow?: () => void;
		onFocus?: () => void;
		focused?: boolean;
	} = $props();

	let host: HTMLDivElement | undefined = $state();
	let view: EditorView | null = null;
	let last_set_value = '';

	const editorTheme = EditorView.theme({
		'&': { fontSize: '12px', color: palette.text, backgroundColor: palette.bgInset, fontFamily: fonts.mono },
		'.cm-scroller': { fontFamily: fonts.mono, lineHeight: '1.55' },
		'.cm-content': { caretColor: palette.accent, padding: '6px 0' },
		'&.cm-focused': { outline: 'none' },
		'.cm-cursor, .cm-dropCursor': { borderLeftColor: palette.accent, borderLeftWidth: '2px' },
		'&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': { background: palette.accentDim },
		'.cm-gutters': { backgroundColor: palette.bgSurface, color: palette.textDim, borderRight: `1px solid ${palette.borderSubtle}`, fontFamily: fonts.mono, fontSize: '10px' },
		'.cm-lineNumbers .cm-gutterElement': { padding: '0 8px 0 10px', minWidth: '32px', color: palette.textDim },
		'.cm-activeLineGutter': { backgroundColor: 'transparent', color: palette.accent },
		'.cm-activeLine': { backgroundColor: palette.bgSurface },
		'.cm-matchingBracket, .cm-nonmatchingBracket': { backgroundColor: palette.accentDim, outline: `1px solid ${palette.accent}`, color: palette.text },
	}, { dark: true });

	const highlight = HighlightStyle.define([
		{ tag: t.keyword, color: palette.accent, fontWeight: '500' },
		{ tag: [t.controlKeyword, t.moduleKeyword], color: palette.accent, fontWeight: '500' },
		{ tag: t.number, color: palette.accentSecondary },
		{ tag: [t.bool, t.null, t.atom], color: palette.accent },
		{ tag: t.string, color: '#6bbf8a' },
		{ tag: t.special(t.string), color: '#7bbf95' },
		{ tag: t.escape, color: palette.accentSecondary },
		{ tag: [t.comment, t.lineComment, t.blockComment], color: palette.textDim, fontStyle: 'italic' },
		{ tag: t.function(t.variableName), color: '#4a9ec2' },
		{ tag: t.function(t.definition(t.variableName)), color: '#4a9ec2', fontWeight: '500' },
		{ tag: t.definition(t.variableName), color: palette.text, fontWeight: '500' },
		{ tag: t.className, color: '#e8944a', fontWeight: '500' },
		{ tag: [t.typeName, t.namespace], color: '#e8944a' },
		{ tag: t.operator, color: palette.textMuted },
		{ tag: [t.punctuation, t.separator], color: palette.textMuted },
		{ tag: [t.brace, t.bracket, t.paren], color: palette.textMuted },
		{ tag: t.variableName, color: palette.text },
		{ tag: t.propertyName, color: palette.text },
		{ tag: [t.self, t.special(t.variableName)], color: palette.accent, fontStyle: 'italic' },
		{ tag: t.meta, color: palette.accentSecondary },
		{ tag: t.invalid, color: palette.accent, textDecoration: 'underline wavy' },
	]);

	function build(initial: string): EditorView {
		const cell_keys = keymap.of([
			{ key: 'Shift-Enter', preventDefault: true, run: () => { onRun(); return true; } },
			{ key: 'Ctrl-Enter', preventDefault: true, run: () => { onRun(); return true; } },
			{ key: 'Mod-Shift-Enter', preventDefault: true, run: () => { onRunAllBelow?.(); return true; } },
		]);
		const extensions: Extension[] = [
			lineNumbers(),
			highlightActiveLine(),
			highlightActiveLineGutter(),
			history(),
			drawSelection(),
			dropCursor(),
			indentOnInput(),
			indentUnit.of('    '),
			bracketMatching(),
			python(),
			syntaxHighlighting(highlight),
			editorTheme,
			cell_keys,
			keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
			EditorView.updateListener.of((upd) => {
				if (upd.docChanged) {
					const text = upd.state.doc.toString();
					last_set_value = text;
					source = text;
				}
				if (upd.focusChanged && upd.view.hasFocus) onFocus?.();
			}),
		];
		return new EditorView({ state: EditorState.create({ doc: initial, extensions }), parent: host! });
	}

	onMount(() => {
		if (!host) return;
		view = build(source);
		last_set_value = source;
	});
	onDestroy(() => view?.destroy());

	$effect(() => {
		if (!view) return;
		if (source === last_set_value) return;
		const current = view.state.doc.toString();
		if (current === source) { last_set_value = source; return; }
		view.dispatch({ changes: { from: 0, to: current.length, insert: source } });
		last_set_value = source;
	});

	// Auto-focus the editor when this cell becomes the focused one
	// (e.g. after a Run-cell advance, or when the Notebook adds a new cell).
	let was_focused = false;
	$effect(() => {
		if (focused && !was_focused && view) view.focus();
		was_focused = focused;
	});

	export function focus() { view?.focus(); }
</script>

<div class="cell" class:focused>
	<div class="cell-head">
		<button class="run" onclick={onRun} disabled={status === 'running'} title="Run cell (Shift+Enter)">
			{#if status === 'running'}
				<svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="6 6"><animateTransform attributeName="transform" type="rotate" from="0 6 6" to="360 6 6" dur="0.9s" repeatCount="indefinite"/></circle></svg>
			{:else}
				<svg width="10" height="10" viewBox="0 0 10 10"><polygon points="2,1 9,5 2,9" fill="currentColor"/></svg>
			{/if}
		</button>
		<span class="idx">In [{index + 1}]</span>
		<span class="status" class:ok={status === 'ok'} class:err={status === 'error'}>
			{#if status === 'ok'}✓{:else if status === 'error'}!{/if}
		</span>
	</div>
	<div class="cell-body" bind:this={host} onfocus={onFocus} onclick={onFocus}></div>
</div>

<style>
	.cell {
		border: 1px solid var(--border-subtle);
		background: var(--bg-inset);
		margin: 0 0 var(--space-md) 0;
		transition: border-color var(--transition);
	}
	.cell.focused { border-color: var(--accent); }
	.cell-head {
		display: flex;
		align-items: center;
		gap: var(--space-md);
		padding: 0 var(--space-md);
		height: 24px;
		background: var(--bg-surface);
		border-bottom: 1px solid var(--border-subtle);
		flex-shrink: 0;
	}
	.run {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 18px;
		height: 18px;
		padding: 0;
		background: transparent;
		border: 1px solid var(--border);
		color: var(--accent);
		cursor: pointer;
		text-transform: none;
		letter-spacing: 0;
		font-weight: normal;
		transition: background var(--transition), border-color var(--transition);
	}
	.run:hover { background: var(--accent-dim); border-color: var(--accent); }
	.run:disabled { cursor: default; opacity: 0.6; }
	.idx {
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
	}
	.status {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-dim);
		font-weight: 700;
		margin-left: auto;
	}
	.status.ok { color: var(--accent); }
	.status.err { color: var(--accent); text-decoration: underline; }
	.cell-body { background: var(--bg-inset); }
</style>
