// Read-only CodeMirror viewer — editor chrome and syntax colours taken
// verbatim from the RapidFEM notebook cell editor so docs code reads
// exactly like a notebook cell.

import { EditorState, type Extension } from '@codemirror/state';
import { EditorView, lineNumbers as cmLineNumbers } from '@codemirror/view';
import { python } from '@codemirror/lang-python';
import { bracketMatching, syntaxHighlighting, HighlightStyle, indentUnit } from '@codemirror/language';
import { tags as t } from '@lezer/highlight';

const c = {
	text: '#e2ddd5',
	textMuted: '#9a96a0',
	textDim: '#8a8790',
	accent: '#d9513c',
	accentSecondary: '#e8944a',
	accentDim: '#d9513c33',
	accentPurple: '#a78bd9',
	bgMid: '#1a1a1f',
	bgSurface: '#232329',
	border: '#35353d',
	borderSubtle: '#2d2d34'
};
const mono = "'JetBrains Mono', monospace";

// Editor chrome — mirrors the notebook cell theme.
const editorTheme = EditorView.theme(
	{
		'&': { fontSize: '12px', color: c.text, backgroundColor: c.bgMid, fontFamily: mono },
		'.cm-scroller': { fontFamily: mono, lineHeight: '1.55' },
		'.cm-content': { padding: '6px 0' },
		'&.cm-focused': { outline: 'none' },
		'.cm-selectionBackground, ::selection': { background: c.accentDim },
		'.cm-gutters': {
			backgroundColor: c.bgSurface,
			color: c.textDim,
			borderRight: `1px solid ${c.borderSubtle}`,
			fontFamily: mono,
			fontSize: '10px'
		},
		'.cm-lineNumbers .cm-gutterElement': {
			padding: '0 8px 0 10px',
			minWidth: '32px',
			color: c.textDim
		}
	},
	{ dark: true }
);

// Syntax colours — mirrors the notebook cell highlight style.
const highlight = HighlightStyle.define([
	{ tag: t.keyword, color: c.accent, fontWeight: '500' },
	{ tag: [t.controlKeyword, t.moduleKeyword], color: c.accent, fontWeight: '500' },
	{ tag: t.number, color: c.accentSecondary },
	{ tag: [t.bool, t.null, t.atom], color: c.accent },
	{ tag: t.string, color: '#6bbf8a' },
	{ tag: t.special(t.string), color: '#7bbf95' },
	{ tag: t.escape, color: c.accentSecondary },
	{ tag: [t.comment, t.lineComment, t.blockComment], color: c.textDim, fontStyle: 'italic' },
	{ tag: t.docComment, color: c.textMuted, fontStyle: 'italic' },
	{ tag: t.function(t.variableName), color: '#4a9ec2' },
	{ tag: t.function(t.definition(t.variableName)), color: '#4a9ec2', fontWeight: '500' },
	{ tag: t.function(t.propertyName), color: c.accentPurple, fontWeight: '500' },
	{ tag: t.definition(t.variableName), color: c.text, fontWeight: '500' },
	{ tag: t.className, color: '#e8944a', fontWeight: '500' },
	{ tag: [t.typeName, t.namespace], color: '#e8944a' },
	{ tag: t.operator, color: c.textMuted },
	{ tag: [t.punctuation, t.separator], color: c.textMuted },
	{ tag: [t.brace, t.bracket, t.paren], color: c.textMuted },
	{ tag: t.variableName, color: c.text },
	{ tag: t.propertyName, color: c.accentPurple },
	{ tag: [t.self, t.special(t.variableName)], color: c.accent, fontStyle: 'italic' },
	{ tag: t.meta, color: c.accentSecondary },
	{ tag: t.invalid, color: c.accent, textDecoration: 'underline wavy' }
]);

/** Create a read-only Python CodeMirror view inside `parent`. */
export function createViewer(
	parent: HTMLElement,
	doc: string,
	options: { lineNumbers?: boolean } = {}
): EditorView {
	const extensions: Extension[] = [
		indentUnit.of('    '),
		bracketMatching(),
		python(),
		syntaxHighlighting(highlight),
		editorTheme,
		EditorState.readOnly.of(true),
		EditorView.editable.of(false)
	];
	if (options.lineNumbers !== false) {
		extensions.unshift(cmLineNumbers());
	}
	return new EditorView({ state: EditorState.create({ doc, extensions }), parent });
}
