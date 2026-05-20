<script lang="ts">
	import { onMount, onDestroy, tick } from 'svelte';
	import type { EditorView } from '@codemirror/view';
	import { createViewer } from '$lib/codemirror';

	// Renders docstring HTML produced by scripts/build.py (docutils RST → HTML).
	// Post-processing: docutils definition lists become parameter tables, and
	// `.math` elements are rendered with KaTeX (loaded lazily on demand).
	interface Props {
		html: string;
	}

	let { html }: Props = $props();

	let container: HTMLDivElement | undefined = $state();
	let rendered = false;

	function transformDefinitionLists() {
		if (!container) return;

		for (const dl of container.querySelectorAll('dl.docutils')) {
			if (dl.classList.contains('table-transformed')) continue;

			const wrapper = document.createElement('div');
			wrapper.className = 'param-table-wrapper';
			const table = document.createElement('table');
			table.className = 'param-table';

			const thead = document.createElement('thead');
			const headerRow = document.createElement('tr');
			for (const text of ['Name', 'Type', 'Description']) {
				const th = document.createElement('th');
				th.textContent = text;
				headerRow.appendChild(th);
			}
			thead.appendChild(headerRow);
			table.appendChild(thead);

			const tbody = document.createElement('tbody');
			for (const dt of dl.querySelectorAll(':scope > dt')) {
				const row = document.createElement('tr');

				const nameCell = document.createElement('td');
				nameCell.className = 'param-name';
				const nameCode = document.createElement('code');
				let name = '';
				for (const node of dt.childNodes) {
					if (node.nodeType === Node.TEXT_NODE) {
						name = node.textContent?.trim() || '';
					} else if (node.nodeType === Node.ELEMENT_NODE) {
						const el = node as Element;
						if (el.classList.contains('classifier-delimiter')) break;
						if (el.classList.contains('classifier')) break;
						name = el.textContent?.trim() || '';
					}
					if (name) break;
				}
				nameCode.textContent = name;
				nameCell.appendChild(nameCode);
				row.appendChild(nameCell);

				const typeCell = document.createElement('td');
				typeCell.className = 'param-type';
				const classifier = dt.querySelector('.classifier');
				if (classifier) {
					const code = document.createElement('code');
					code.textContent = classifier.textContent || '';
					typeCell.appendChild(code);
				}
				row.appendChild(typeCell);

				const descCell = document.createElement('td');
				descCell.className = 'param-desc';
				const dd = dt.nextElementSibling;
				if (dd && dd.tagName === 'DD') descCell.innerHTML = dd.innerHTML;
				row.appendChild(descCell);

				tbody.appendChild(row);
			}
			table.appendChild(tbody);
			wrapper.appendChild(table);
			dl.parentNode?.replaceChild(wrapper, dl);
		}
	}

	async function renderMath() {
		if (!container) return;

		// Block math: docutils `.. math::` → <pre class="math"> / <div class="math">.
		const blocks = [...container.querySelectorAll('.math')].filter(
			(el) => !el.classList.contains('katex-rendered')
		);
		// Inline math authored as RST literals of raw LaTeX (\varepsilon_\infty, \tau …).
		const inlines = [...container.querySelectorAll('tt, code')].filter(
			(el) => !el.closest('pre') && /^\\[a-zA-Z]/.test((el.textContent || '').trim())
		);
		if (blocks.length === 0 && inlines.length === 0) return;

		const katex = (await import('katex')).default;
		await import('katex/dist/katex.min.css');

		for (const el of blocks) {
			const latex = (el.textContent || '').trim();
			if (!latex) continue;
			try {
				let cleaned = latex.replace(/^\\\(|\\\)$/g, '').replace(/^\\\[|\\\]$/g, '').trim();
				cleaned = cleaned
					.replace(/\\begin\{eqnarray\*?\}/g, '\\begin{aligned}')
					.replace(/\\end\{eqnarray\*?\}/g, '\\end{aligned}');
				if (cleaned.includes('\\\\') && !cleaned.includes('\\begin{')) {
					cleaned = `\\begin{aligned}${cleaned}\\end{aligned}`;
				}
				const isDisplay = el.tagName === 'DIV' || el.tagName === 'PRE';
				el.innerHTML = katex.renderToString(cleaned, {
					displayMode: isDisplay,
					throwOnError: false,
					strict: false
				});
				el.classList.add('katex-rendered');
			} catch {
				/* leave raw LaTeX in place */
			}
		}

		for (const el of inlines) {
			const tex = (el.textContent || '').trim();
			try {
				el.innerHTML = katex.renderToString(tex, {
					displayMode: false,
					throwOnError: false,
					strict: false
				});
				el.classList.add('katex-inline');
			} catch {
				/* leave as literal */
			}
		}
	}

	// Replace docstring <pre> code blocks with read-only CodeMirror viewers
	// so example code reads exactly like a notebook cell.
	let codeViews: EditorView[] = [];

	function renderCodeBlocks() {
		if (!container) return;
		for (const pre of container.querySelectorAll('pre')) {
			if (pre.classList.contains('cm-done')) continue;
			// Math blocks are handled by renderMath / KaTeX — never CodeMirror.
			if (pre.classList.contains('math')) continue;
			const code = (pre.textContent || '').replace(/\n$/, '');
			if (!code.trim()) continue;

			const wrapper = document.createElement('div');
			wrapper.className = 'docstring-code';
			pre.parentNode?.replaceChild(wrapper, pre);
			wrapper.classList.add('cm-done');
			codeViews.push(createViewer(wrapper, code, { lineNumbers: false }));
		}
	}

	onMount(() => {
		if (html && container && !rendered) {
			rendered = true;
			tick().then(() => {
				transformDefinitionLists();
				renderCodeBlocks();
				renderMath();
			});
		}
	});

	onDestroy(() => {
		for (const view of codeViews) view.destroy();
		codeViews = [];
	});
</script>

<div class="docstring-content" bind:this={container}>
	{@html html}
</div>

<style>
	.docstring-content {
		font-size: var(--fs-sm);
		line-height: 1.7;
		color: var(--text-muted);
	}

	.docstring-content :global(p) {
		margin-bottom: 0.75em;
	}

	.docstring-content :global(p:last-child) {
		margin-bottom: 0;
	}

	/* Parameter / attribute tables */
	.docstring-content :global(.param-table-wrapper) {
		margin: var(--space-md) 0;
	}

	.docstring-content :global(.param-table) {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--fs-xs);
	}

	.docstring-content :global(.param-table thead th) {
		padding: var(--space-xs) var(--space-md);
		font-weight: 600;
		color: var(--accent);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		text-align: left;
		border-bottom: 1px solid var(--border);
	}

	.docstring-content :global(.param-table td) {
		padding: var(--space-sm) var(--space-md);
		vertical-align: top;
		border-bottom: 1px solid var(--border-subtle);
	}

	.docstring-content :global(.param-table tbody tr:last-child td) {
		border-bottom: 1px solid var(--border);
	}

	.docstring-content :global(.param-table .param-name code) {
		font-family: var(--font-mono);
		color: var(--accent);
		background: none;
		border: none;
		padding: 0;
		white-space: nowrap;
	}

	.docstring-content :global(.param-table .param-type code) {
		font-family: var(--font-mono);
		color: var(--text-muted);
		background: none;
		border: none;
		padding: 0;
	}

	.docstring-content :global(.param-table .param-desc) {
		color: var(--text-muted);
		line-height: 1.5;
	}

	/* Lists */
	.docstring-content :global(ul),
	.docstring-content :global(ol) {
		margin: var(--space-sm) 0;
		padding-left: var(--space-xl);
	}

	.docstring-content :global(li) {
		margin-bottom: var(--space-xs);
	}

	/* Section headers (NumPy sections become <p><strong>) */
	.docstring-content :global(h3),
	.docstring-content :global(h4),
	.docstring-content :global(p:has(> strong:only-child)) {
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		font-weight: 600;
		color: var(--accent);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin: var(--space-lg) 0 var(--space-sm);
		padding: 0;
		border: none;
	}

	.docstring-content :global(p:first-child:has(> strong:only-child)) {
		margin-top: 0;
	}

	.docstring-content :global(strong) {
		font-weight: 600;
	}

	/* Code blocks inside docstrings — CodeMirror viewer */
	.docstring-content :global(.docstring-code) {
		margin: var(--space-md) 0;
		border: 1px solid var(--border);
		background: var(--surface-mid);
	}

	/* Fallback for any <pre> not yet upgraded to CodeMirror */
	.docstring-content :global(pre) {
		background: var(--surface-inset);
		border: 1px solid var(--border);
		padding: var(--space-md);
		margin: var(--space-md) 0;
		overflow-x: auto;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		color: var(--text);
		line-height: 1.55;
	}

	/* Inline code / literals — no backdrop or border, just colour. */
	.docstring-content :global(code),
	.docstring-content :global(tt) {
		font-family: var(--font-mono);
		color: var(--accent-secondary);
	}

	.docstring-content :global(blockquote) {
		margin: var(--space-md) 0;
		padding-left: var(--space-md);
		border-left: 3px solid var(--accent);
		color: var(--text-muted);
	}

	.docstring-content :global(table:not(.param-table)) {
		width: 100%;
		border-collapse: collapse;
		margin: var(--space-md) 0;
	}

	.docstring-content :global(table:not(.param-table) th),
	.docstring-content :global(table:not(.param-table) td) {
		padding: var(--space-xs) var(--space-sm);
		border: 1px solid var(--border);
		text-align: left;
	}

	.docstring-content :global(.katex-display) {
		margin: var(--space-md) 0;
	}

	/* Math blocks — no backdrop, border or scroll container. Same muted
	   colour as the surrounding docstring text. */
	.docstring-content :global(pre.math),
	.docstring-content :global(div.math) {
		background: none;
		border: none;
		padding: 0;
		margin: var(--space-md) 0;
		color: var(--text-muted);
		white-space: normal;
	}

	/* Inline math rendered from LaTeX literals — muted, like the body text. */
	.docstring-content :global(.katex-inline) {
		color: var(--text-muted);
	}

	.docstring-content :global(.math:not(.katex-rendered)) {
		font-family: var(--font-mono);
		color: var(--text-muted);
	}
</style>
