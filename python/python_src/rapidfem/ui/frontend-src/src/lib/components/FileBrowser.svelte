<script lang="ts">
	import { listFiles, listExamples, deleteFile, renameFile, type FileEntry } from '$lib/api';

	let {
		active_path = $bindable<string | null>(null),
		onOpen,
		onNew,
		onOpenExample,
		onClosed,
	}: {
		active_path: string | null;
		onOpen: (path: string) => void;
		onNew: () => void;
		onOpenExample?: (name: string) => void;
		onClosed?: (path: string) => void;
	} = $props();

	let files = $state<FileEntry[]>([]);
	let examples = $state<{ name: string }[]>([]);
	let workdir = $state('');
	let error = $state<string | null>(null);
	let loading = $state(false);

	async function refresh() {
		loading = true;
		try {
			const [r, ex] = await Promise.all([listFiles(), listExamples()]);
			workdir = r.workdir;
			files = r.files;
			examples = ex.examples;
			error = null;
		} catch (e) {
			error = String(e);
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		void refresh();
	});

	function nice_path(p: string): string {
		return p.replace(/\\/g, '/');
	}

	async function on_delete(e: MouseEvent, path: string) {
		e.stopPropagation();
		if (!confirm(`Delete ${path}?`)) return;
		try {
			await deleteFile(path);
			if (active_path === path) {
				active_path = null;
				onClosed?.(path);
			}
			await refresh();
		} catch (err) {
			error = String(err);
		}
	}

	async function on_rename(e: MouseEvent, path: string) {
		e.stopPropagation();
		const next = prompt('Rename to:', path);
		if (!next || next === path) return;
		try {
			await renameFile(path, next);
			if (active_path === path) {
				active_path = next;
				onOpen(next);
			}
			await refresh();
		} catch (err) {
			error = String(err);
		}
	}
</script>

<div class="browser">
	<div class="head">
		<span class="title">Files</span>
		<button class="tb" onclick={onNew} title="New .py file" aria-label="New file">
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
				<path d="M8 3v10" />
				<path d="M3 8h10" />
			</svg>
		</button>
		<button class="tb" onclick={refresh} title="Refresh" aria-label="Refresh" disabled={loading}>
			<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
				<path d="M2.5 8a5.5 5.5 0 0 1 9.5-3.8" />
				<polyline points="12.5,2 12.5,4.5 10,4.5" />
				<path d="M13.5 8a5.5 5.5 0 0 1-9.5 3.8" />
				<polyline points="3.5,14 3.5,11.5 6,11.5" />
			</svg>
		</button>
	</div>
	{#if error}
		<div class="error">{error}</div>
	{/if}
	<div class="list">
		<div class="section">Workdir</div>
		{#each files as f (f.path)}
			<!-- svelte-ignore a11y_click_events_have_key_events a11y_no_static_element_interactions -->
			<div class="item-row" class:active={f.path === active_path} ondblclick={(e) => on_rename(e, f.path)}>
				<button
					class="item"
					onclick={() => onOpen(f.path)}
					title={`${f.path}  ·  ${f.size} bytes  ·  double-click row to rename`}
				>
					<span class="name">{nice_path(f.path)}</span>
				</button>
				<button class="row-act" onclick={(e) => on_delete(e, f.path)} title="Delete" aria-label="Delete">
					<svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
						<polyline points="3,5 13,5" />
						<path d="M6 5V3h4v2" />
						<path d="M5 5l1 8h4l1-8" />
					</svg>
				</button>
			</div>
		{/each}
		{#if !files.length && !loading}
			<div class="empty">No .py files yet.</div>
		{/if}

		{#if examples.length}
			<div class="section">Examples</div>
			{#each examples as e (e.name)}
				<button
					class="item example"
					onclick={() => onOpenExample?.(e.name)}
					title={`Bundled example — ${e.name}`}
				>
					<span class="name">{e.name}</span>
				</button>
			{/each}
		{/if}
	</div>
</div>

<style>
	.browser {
		display: flex;
		flex-direction: column;
		height: 100%;
		background: var(--bg-surface);
		color: var(--text);
	}
	.head {
		display: flex;
		align-items: center;
		gap: var(--space-sm);
		padding: 0 var(--space-lg);
		height: 36px;
		flex-shrink: 0;
		border-bottom: 1px solid var(--border);
		background: var(--bg-surface);
	}
	.head .title {
		flex: 1;
		color: var(--text-muted);
		text-transform: uppercase;
		font-size: var(--fs-xs);
		letter-spacing: 0.5px;
		font-weight: 600;
	}
	.tb {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 24px;
		height: 24px;
		padding: 0;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		color: var(--text-muted);
		cursor: pointer;
		transition: background var(--transition), border-color var(--transition), color var(--transition);
	}
	.tb:hover {
		background: var(--bg-panel);
		border-color: var(--accent);
		color: var(--text);
	}
	.tb:disabled {
		opacity: 0.4;
		cursor: default;
		background: var(--bg-surface);
		border-color: var(--border);
		color: var(--text-dim);
	}
	.tb svg { display: block; }
	.list { flex: 1; overflow: auto; padding: var(--space-sm) 0; }
	.section {
		padding: var(--space-md) var(--space-lg) var(--space-xs);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		color: var(--text-dim);
		font-weight: 600;
	}
	.item.example { color: var(--text-dim); font-style: italic; }
	.item.example:hover { background: var(--bg-panel); color: var(--accent-secondary); }
	.item {
		display: block;
		width: 100%;
		text-align: left;
		background: transparent;
		border: 0;
		color: var(--text-muted);
		padding: 4px var(--space-lg);
		cursor: pointer;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		text-transform: none;
		letter-spacing: 0;
		font-weight: 400;
	}
	.item:hover {
		background: var(--bg-panel);
		color: var(--text);
	}
	.item-row {
		display: flex;
		align-items: center;
		position: relative;
	}
	.item-row.active {
		background: var(--accent-dim);
		border-left: 2px solid var(--accent);
	}
	.item-row.active .item { color: var(--accent); padding-left: calc(var(--space-lg) - 2px); }
	.item .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; flex: 1; }
	.item-row .item { flex: 1; }
	.row-act {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 16px;
		height: 16px;
		padding: 0;
		margin-right: var(--space-md);
		background: transparent;
		border: 0;
		color: transparent;
		cursor: pointer;
		text-transform: none;
		letter-spacing: 0;
		font-weight: normal;
		flex-shrink: 0;
		transition: color var(--transition);
	}
	.item-row:hover .row-act { color: var(--text-dim); }
	.row-act:hover { color: var(--text-muted); }
	.empty { color: var(--text-dim); padding: var(--space-md) var(--space-lg); font-style: italic; font-size: var(--fs-xs); }
	.error { color: var(--accent); padding: var(--space-md) var(--space-lg); font-size: var(--fs-xs); }
</style>
