<script lang="ts">
	import { onMount } from 'svelte';
	import Cell from './Cell.svelte';

	let {
		source = $bindable<string>(''),
		file_path = '',
		onRunCell,
		onRunAll,
		onResetKernel,
	}: {
		source: string;
		file_path: string;
		onRunCell: (cell_source: string, reset_first: boolean) => Promise<'ok' | 'error'>;
		onRunAll?: () => void;
		onResetKernel?: () => void;
	} = $props();

	// Cells = blocks of source text separated by `# %%` markers at the
	// start of a line. Each cell includes its leading marker line so we
	// can reassemble the file byte-perfect on save.
	type Cell = {
		id: number;
		text: string;        // contents of the cell (without trailing newline)
		marker: string | null; // the "# %%" line, or null for the implicit first cell
		status: 'idle' | 'running' | 'ok' | 'error';
	};

	let cells = $state<Cell[]>([]);
	let next_id = 1;
	let focused_id = $state<number | null>(null);
	let cell_refs: Map<number, ReturnType<typeof Cell>> = new Map();

	function parse(text: string): Cell[] {
		// Split keeping the marker lines. A marker line is one that, after
		// optional whitespace, starts with `# %%`.
		const lines = text.split('\n');
		const out: Cell[] = [];
		let buf: string[] = [];
		let marker: string | null = null;
		const flush = () => {
			out.push({ id: next_id++, text: buf.join('\n'), marker, status: 'idle' });
		};
		for (const line of lines) {
			if (/^\s*#\s*%%/.test(line)) {
				flush();
				marker = line;
				buf = [];
			} else {
				buf.push(line);
			}
		}
		flush();
		// Drop leading empty cell if file starts with a marker
		if (out.length > 1 && out[0].marker === null && out[0].text.trim() === '') {
			out.shift();
		}
		return out;
	}

	function serialize(): string {
		const parts: string[] = [];
		for (const c of cells) {
			if (c.marker !== null) parts.push(c.marker);
			parts.push(c.text);
		}
		return parts.join('\n');
	}

	let last_source = '';
	$effect(() => {
		if (source === last_source) return;
		last_source = source;
		cells = parse(source);
		focused_id = cells[0]?.id ?? null;
	});

	// Push edits back up. The reverse effect only fires if cells changed in
	// a way that produces different text — small noise from `parse` round-
	// trips is OK since we compare strings.
	function push_source() {
		const t = serialize();
		if (t === source) return;
		last_source = t;
		source = t;
	}

	function is_md(c: Cell): boolean {
		return !!c.marker && /\[markdown\]/i.test(c.marker);
	}

	async function run_cell(c: Cell, opts: { reset?: boolean; advance?: boolean } = {}) {
		if (!is_md(c)) {
			c.status = 'running';
			cells = [...cells];
			try {
				const result = await onRunCell(c.text, !!opts.reset);
				c.status = result;
			} catch {
				c.status = 'error';
			}
			cells = [...cells];
		}
		// Jupyter-style: after running a cell, move the focus to the next
		// cell (or insert a fresh one at the bottom).
		if (opts.advance !== false) {
			const i = cells.findIndex((x) => x.id === c.id);
			if (i >= 0 && i < cells.length - 1) {
				focused_id = cells[i + 1].id;
			} else if (i === cells.length - 1 && !is_md(c)) {
				add_cell_after(c.id);
			}
		}
	}

	async function run_all() {
		let first = true;
		for (const c of cells) {
			if (is_md(c)) continue;
			await run_cell(c, { reset: first, advance: false });
			first = false;
			if (c.status === 'error') break;
		}
	}

	async function run_focused() {
		const c = cells.find((c) => c.id === focused_id);
		if (c) await run_cell(c, { advance: true });
	}

	function add_cell_after(id: number) {
		const i = cells.findIndex((c) => c.id === id);
		if (i < 0) return;
		const fresh: Cell = { id: next_id++, text: '', marker: '# %%', status: 'idle' };
		cells = [...cells.slice(0, i + 1), fresh, ...cells.slice(i + 1)];
		focused_id = fresh.id;
		push_source();
	}

	function delete_cell(id: number) {
		if (cells.length <= 1) return;
		cells = cells.filter((c) => c.id !== id);
		push_source();
	}

	export function run_all_cells() { void run_all(); }
	export function run_current_cell() { void run_focused(); }
	export function add_cell() {
		if (focused_id != null) add_cell_after(focused_id);
		else if (cells.length) add_cell_after(cells[cells.length - 1].id);
	}
</script>

<div class="notebook">
	{#each cells as cell (cell.id)}
		{@const is_md = !!cell.marker && /\[markdown\]/i.test(cell.marker)}
		<Cell
			index={cells.indexOf(cell)}
			bind:source={cell.text}
			status={cell.status}
			kind={is_md ? 'markdown' : 'code'}
			focused={cell.id === focused_id}
			onRun={() => { focused_id = cell.id; void run_cell(cell, { advance: true }); }}
			onRunAllBelow={() => { focused_id = cell.id; void run_all(); }}
			onFocus={() => { focused_id = cell.id; }}
		/>
	{/each}
	{#if cells.length === 0}
		<div class="empty">Open a file or pick an example to start.</div>
	{/if}
	{#if cells.length > 0}
		<button
			class="add-cell"
			onclick={() => { if (focused_id != null) add_cell_after(focused_id); }}
			title="Add cell below"
		>
			+ Add cell
		</button>
	{/if}
</div>

<style>
	.notebook {
		height: 100%;
		overflow: auto;
		padding: var(--space-lg);
		background: var(--bg);
	}
	.empty {
		color: var(--text-dim);
		font-style: italic;
		text-align: center;
		padding: var(--space-3xl);
		font-size: var(--fs-sm);
	}
	.add-cell {
		display: block;
		width: 100%;
		background: transparent;
		border: 1px dashed var(--border);
		color: var(--text-dim);
		padding: 6px;
		cursor: pointer;
		text-transform: none;
		letter-spacing: 0;
		font-weight: normal;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		transition: color var(--transition), border-color var(--transition);
	}
	.add-cell:hover { color: var(--accent); border-color: var(--accent); }
</style>
