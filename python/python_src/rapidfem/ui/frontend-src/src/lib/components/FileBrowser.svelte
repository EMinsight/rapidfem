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
	.browser { display: flex; flex-direction: column; height: 100%; background: #131313; color: #ccc; font-size: 12px; }
	.head { display: flex; align-items: center; gap: 4px; padding: 6px 8px; border-bottom: 1px solid #2a2a2a; }
	.head .title { flex: 1; color: #aaa; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }
	.head button { background: transparent; border: 1px solid #333; color: #ccc; padding: 2px 6px; cursor: pointer; font-size: 12px; }
	.head button:disabled { opacity: 0.4; cursor: default; }
	.list { flex: 1; overflow: auto; padding: 2px 0; }
	.item { display: block; width: 100%; text-align: left; background: transparent; border: 0; color: #ccc; padding: 3px 10px; cursor: pointer; font: 12px ui-monospace, Consolas, monospace; }
	.item:hover { background: #1f1f1f; }
	.item.active { background: #243a36; color: #fff; }
	.item .name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; }
	.empty { color: #666; padding: 8px 10px; font-style: italic; }
	.error { color: #d77; padding: 6px 10px; }
</style>
