<script lang="ts">
	import { listFiles, type FileEntry } from '$lib/api';

	let {
		active_path = $bindable<string | null>(null),
		onOpen,
		onNew,
	}: {
		active_path: string | null;
		onOpen: (path: string) => void;
		onNew: () => void;
	} = $props();

	let files = $state<FileEntry[]>([]);
	let workdir = $state('');
	let error = $state<string | null>(null);
	let loading = $state(false);

	async function refresh() {
		loading = true;
		try {
			const r = await listFiles();
			workdir = r.workdir;
			files = r.files;
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
</script>

<div class="browser">
	<div class="head">
		<span class="title">Files</span>
		<button onclick={onNew} title="New .py file">＋</button>
		<button onclick={refresh} title="Refresh" disabled={loading}>⟳</button>
	</div>
	{#if error}
		<div class="error">{error}</div>
	{/if}
	<div class="list">
		{#each files as f (f.path)}
			<button
				class:active={f.path === active_path}
				class="item"
				onclick={() => onOpen(f.path)}
				title={`${f.path}  ·  ${f.size} bytes`}
			>
				<span class="name">{nice_path(f.path)}</span>
			</button>
		{/each}
		{#if !files.length && !loading}
			<div class="empty">No .py files in workdir.</div>
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
		padding: var(--space-md) var(--space-lg);
		min-height: 38px;
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
	.head button {
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text-muted);
		padding: 0;
		width: 22px;
		height: 22px;
		cursor: pointer;
		font-size: var(--fs-md);
		line-height: 1;
		text-transform: none;
		letter-spacing: 0;
	}
	.head button:hover {
		background: var(--bg-panel);
		color: var(--text);
		border-color: var(--input-hover);
	}
	.head button:disabled { opacity: 0.4; cursor: default; }
	.list { flex: 1; overflow: auto; padding: var(--space-sm) 0; }
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
		font-size: var(--fs-xs);
		text-transform: none;
		letter-spacing: 0;
		font-weight: 400;
	}
	.item:hover {
		background: var(--bg-panel);
		color: var(--text);
	}
	.item.active {
		background: var(--accent-dim);
		color: var(--accent);
		border-left: 2px solid var(--accent);
		padding-left: calc(var(--space-lg) - 2px);
	}
	.item .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
	.empty { color: var(--text-dim); padding: var(--space-md) var(--space-lg); font-style: italic; font-size: var(--fs-xs); }
	.error { color: var(--accent); padding: var(--space-md) var(--space-lg); font-size: var(--fs-xs); }
</style>
