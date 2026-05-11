<script lang="ts">
	import { listFiles, type FileEntry } from '$lib/api';

	let {
		active_path = $bindable<string | null>(null),
		onOpen,
		onNew,
		onCollapse,
	}: {
		active_path: string | null;
		onOpen: (path: string) => void;
		onNew: () => void;
		onCollapse?: () => void;
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
		{#if onCollapse}
			<button class="tb" onclick={onCollapse} title="Collapse" aria-label="Collapse">
				<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
					<polyline points="10,3 5,8 10,13" />
				</svg>
			</button>
		{/if}
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
