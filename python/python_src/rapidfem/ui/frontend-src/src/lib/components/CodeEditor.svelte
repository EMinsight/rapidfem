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
		value = $bindable<string>(''),
		onSave,
	}: {
		value: string;
		onSave?: (text: string) => void;
	} = $props();

	let host: HTMLDivElement | undefined = $state();
	let view: EditorView | null = null;
	let last_set_value = '';

	// ── Theme: editor chrome ────────────────────────────────────────────────
	const editorTheme = EditorView.theme(
		{
			'&': {
				height: '100%',
				fontSize: '13px',
				color: palette.text,
				backgroundColor: palette.bgInset,
				fontFamily: fonts.mono,
			},
			'.cm-scroller': {
				fontFamily: fonts.mono,
				lineHeight: '1.55',
			},
			'.cm-content': {
				caretColor: palette.accent,
				padding: '8px 0',
			},
			'&.cm-focused': { outline: 'none' },
			'.cm-cursor, .cm-dropCursor': { borderLeftColor: palette.accent, borderLeftWidth: '2px' },
			'&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': {
				background: palette.accentDim,
			},
			'.cm-gutters': {
				backgroundColor: palette.bgSurface,
				color: palette.textDim,
				borderRight: `1px solid ${palette.borderSubtle}`,
				fontFamily: fonts.mono,
				fontSize: '11px',
				userSelect: 'none',
			},
			'.cm-lineNumbers .cm-gutterElement': {
				padding: '0 10px 0 12px',
				minWidth: '36px',
				color: palette.textDim,
			},
			'.cm-activeLineGutter': {
				backgroundColor: 'transparent',
				color: palette.accent,
			},
			'.cm-activeLine': {
				backgroundColor: palette.bgSurface,
			},
			'.cm-matchingBracket, .cm-nonmatchingBracket': {
				backgroundColor: palette.accentDim,
				outline: `1px solid ${palette.accent}`,
				color: palette.text,
			},
			'.cm-tooltip': {
				backgroundColor: palette.bgSurface,
				border: `1px solid ${palette.border}`,
				color: palette.text,
				fontFamily: fonts.mono,
			},
		},
		{ dark: true },
	);

	// ── Theme: syntax colors ────────────────────────────────────────────────
	const highlight = HighlightStyle.define([
		// Keywords (def, class, if, for, import, return, …) + control flow
		{ tag: t.keyword, color: palette.accent, fontWeight: '500' },
		{ tag: [t.controlKeyword, t.moduleKeyword], color: palette.accent, fontWeight: '500' },
		// Literals: numbers in the warm secondary; bool/None get the accent
		{ tag: t.number, color: palette.accentSecondary },
		{ tag: [t.bool, t.null, t.atom], color: palette.accent },
		// Strings: cool green to read as data
		{ tag: t.string, color: '#6bbf8a' },
		{ tag: t.special(t.string), color: '#7bbf95' },
		{ tag: t.escape, color: palette.accentSecondary },
		// Comments: dimmed + italic
		{ tag: [t.comment, t.lineComment, t.blockComment], color: palette.textDim, fontStyle: 'italic' },
		{ tag: t.docComment, color: palette.textMuted, fontStyle: 'italic' },
		// Function definitions + calls
		{ tag: t.function(t.variableName), color: '#4a9ec2' },
		{ tag: t.function(t.definition(t.variableName)), color: '#4a9ec2', fontWeight: '500' },
		{ tag: t.definition(t.variableName), color: palette.text, fontWeight: '500' },
		// Class names + types
		{ tag: t.className, color: '#e8944a', fontWeight: '500' },
		{ tag: [t.typeName, t.namespace], color: '#e8944a' },
		// Operators, punctuation, separators
		{ tag: t.operator, color: palette.textMuted },
		{ tag: [t.punctuation, t.separator], color: palette.textMuted },
		{ tag: [t.brace, t.bracket, t.paren], color: palette.textMuted },
		// Variables fall back to body text
		{ tag: t.variableName, color: palette.text },
		{ tag: t.propertyName, color: palette.text },
		{ tag: [t.self, t.special(t.variableName)], color: palette.accent, fontStyle: 'italic' },
		// Decorators (@…)
		{ tag: t.meta, color: palette.accentSecondary },
		// Invalid / errors
		{ tag: t.invalid, color: palette.accent, textDecoration: 'underline wavy' },
	]);

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
			save_keymap,
			keymap.of([...defaultKeymap, ...historyKeymap, indentWithTab]),
			EditorView.updateListener.of((upd) => {
				if (upd.docChanged) {
					const text = upd.state.doc.toString();
					last_set_value = text;
					value = text;
				}
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
		background: var(--bg-inset);
		overflow: hidden;
	}
</style>
