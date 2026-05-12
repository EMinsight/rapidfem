<script lang="ts">
	import { onDestroy } from 'svelte';
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
		kind = 'code' as 'code' | 'markdown',
		readonly = false,
		onRun,
		onRunAllBelow,
		onFocus,
		focused = false,
	}: {
		index: number;
		source: string;
		status?: 'idle' | 'running' | 'ok' | 'error';
		kind?: 'code' | 'markdown';
		readonly?: boolean;
		onRun: () => void;
		onRunAllBelow?: () => void;
		onFocus?: () => void;
		focused?: boolean;
	} = $props();

	let edit_mode = $state(false);

	// Minimal markdown rendering: strip leading "# " from each line (Python
	// comment hash) and translate #/##/### headings + **bold**/*italic*/`code`.
	function render_md(text: string): string {
		const lines = text.split('\n').map((l) => l.replace(/^\s*#\s?/, ''));
		const blocks: string[] = [];
		let para: string[] = [];
		const flush_para = () => {
			if (!para.length) return;
			blocks.push('<p>' + inline(para.join(' ')) + '</p>');
			para = [];
		};
		for (const l of lines) {
			const t = l.trim();
			if (!t) { flush_para(); continue; }
			let m;
			if ((m = t.match(/^(#{1,4})\s+(.*)$/))) {
				flush_para();
				const lvl = m[1].length;
				blocks.push(`<h${lvl}>${inline(m[2])}</h${lvl}>`);
			} else if (/^[-*]\s+/.test(t)) {
				flush_para();
				blocks.push('<li>' + inline(t.replace(/^[-*]\s+/, '')) + '</li>');
			} else {
				para.push(t);
			}
		}
		flush_para();
		// Wrap consecutive <li> in <ul>
		return blocks.join('\n')
			.replace(/(?:<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);
	}
	function inline(s: string): string {
		// Escape first, then re-introduce intended tags
		const esc = s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
		return esc
			.replace(/`([^`]+)`/g, '<code>$1</code>')
			.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
			.replace(/\b_([^_]+)_\b/g, '<em>$1</em>');
	}

	const rendered_md = $derived(kind === 'markdown' ? render_md(source) : '');

	let host: HTMLDivElement | undefined = $state();
	let view: EditorView | null = null;
	let last_set_value = '';

	const editorTheme = EditorView.theme({
		'&': { fontSize: '12px', color: palette.text, backgroundColor: palette.bgMid, fontFamily: fonts.mono },
		'.cm-scroller': { fontFamily: fonts.mono, lineHeight: '1.55' },
		'.cm-content': { caretColor: palette.accent, padding: '6px 0' },
		'&.cm-focused': { outline: 'none' },
		'.cm-cursor, .cm-dropCursor': { borderLeftColor: palette.accent, borderLeftWidth: '2px' },
		'&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection': { background: palette.accentDim },
		'.cm-gutters': { backgroundColor: palette.bgSurface, color: palette.textDim, borderRight: `1px solid ${palette.borderSubtle}`, fontFamily: fonts.mono, fontSize: '10px' },
		'.cm-lineNumbers .cm-gutterElement': { padding: '0 8px 0 10px', minWidth: '32px', color: palette.textDim },
		'.cm-activeLineGutter': { backgroundColor: 'transparent', color: palette.accent },
		'.cm-activeLine': { backgroundColor: 'rgba(255,255,255,0.025)' },
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
		{ tag: t.function(t.propertyName), color: palette.accentPurple, fontWeight: '500' },
		{ tag: t.definition(t.variableName), color: palette.text, fontWeight: '500' },
		{ tag: t.className, color: '#e8944a', fontWeight: '500' },
		{ tag: [t.typeName, t.namespace], color: '#e8944a' },
		{ tag: t.operator, color: palette.textMuted },
		{ tag: [t.punctuation, t.separator], color: palette.textMuted },
		{ tag: [t.brace, t.bracket, t.paren], color: palette.textMuted },
		{ tag: t.variableName, color: palette.text },
		{ tag: t.propertyName, color: palette.accentPurple },
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
			...(readonly ? [EditorState.readOnly.of(true)] : []),
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

	$effect(() => {
		// Build (or rebuild) the CodeMirror editor when we enter code/edit mode.
		const want_editor = kind === 'code' || edit_mode;
		if (!host) return;
		if (want_editor && !view) {
			view = build(source);
			last_set_value = source;
		} else if (!want_editor && view) {
			view.destroy();
			view = null;
		}
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

	let cell_root: HTMLElement | undefined = $state();

	// Auto-focus + scroll-into-view when this cell becomes the focused one
	// (e.g. after a Run-cell advance, or when the Notebook adds a new cell).
	let was_focused = false;
	$effect(() => {
		if (focused && !was_focused) {
			view?.focus();
			cell_root?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
		}
		was_focused = focused;
	});

	export function focus() {
		view?.focus();
		cell_root?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
	}
</script>

<div class="cell" class:focused class:markdown={kind === 'markdown'} bind:this={cell_root}>
	<div class="cell-head">
		{#if kind === 'code'}
			<button class="run" onclick={onRun} disabled={status === 'running' || readonly} title={readonly ? 'Run disabled (static demo)' : 'Run cell (Shift+Enter)'}>
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
		{:else}
			<span class="kind-tag">MD</span>
			<span class="idx">Markdown</span>
			{#if edit_mode}
				<button class="md-toggle" onclick={() => (edit_mode = false)} title="Stop editing">Done</button>
			{/if}
		{/if}
	</div>
	{#if kind === 'markdown' && !edit_mode}
		<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
		<div class="md-rendered" ondblclick={() => { edit_mode = true; onFocus?.(); }} onclick={onFocus} title="Double-click to edit">
			{@html rendered_md}
		</div>
	{:else}
		<div class="cell-body" bind:this={host} onfocus={onFocus} onclick={onFocus}></div>
	{/if}
</div>

<style>
	.cell {
		border: 1px solid var(--border-subtle);
		background: var(--bg-mid);
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
	.cell-body { background: var(--bg-mid); }

	.cell.markdown { border: 0; background: transparent; }
	.cell.markdown .cell-head { background: transparent; border: 0; height: 18px; padding-left: 0; }
	.kind-tag {
		font-family: var(--font-mono);
		font-size: 9px;
		color: var(--text-dim);
		background: var(--bg-surface);
		padding: 1px 5px;
		border: 1px solid var(--border-subtle);
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.md-toggle {
		margin-left: auto;
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text-muted);
		padding: 0 var(--space-md);
		height: 18px;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		cursor: pointer;
	}
	.md-toggle:hover { color: var(--accent); border-color: var(--accent); }

	.md-rendered {
		padding: var(--space-sm) var(--space-md) var(--space-lg);
		color: var(--text);
		font-family: var(--font-body);
		font-size: var(--fs-sm);
		line-height: 1.55;
		cursor: text;
	}
	.md-rendered :global(h1) {
		font-size: 22px; font-weight: 600; color: var(--accent);
		margin: 0 0 var(--space-md); border-bottom: 1px solid var(--border-subtle); padding-bottom: 4px;
	}
	.md-rendered :global(h2) { font-size: 18px; font-weight: 600; color: var(--text); margin: var(--space-md) 0 var(--space-sm); }
	.md-rendered :global(h3) { font-size: 15px; font-weight: 600; color: var(--text); margin: var(--space-md) 0 var(--space-sm); }
	.md-rendered :global(h4) { font-size: 13px; font-weight: 600; color: var(--text-muted); margin: var(--space-md) 0 var(--space-sm); }
	.md-rendered :global(p) { margin: 0 0 var(--space-sm); color: var(--text-muted); }
	.md-rendered :global(ul) { margin: 0 0 var(--space-sm) var(--space-lg); padding: 0; color: var(--text-muted); }
	.md-rendered :global(li) { margin-bottom: 2px; }
	.md-rendered :global(code) {
		font-family: var(--font-mono);
		font-size: 11px;
		background: var(--bg-surface);
		padding: 0 4px;
		border: 1px solid var(--border-subtle);
		color: var(--accent-secondary);
	}
	.md-rendered :global(strong) { color: var(--text); }
	.md-rendered :global(em) { font-style: italic; color: var(--text); }
</style>
