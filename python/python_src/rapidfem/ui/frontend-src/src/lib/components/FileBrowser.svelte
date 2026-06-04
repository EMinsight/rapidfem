<script lang="ts">
	import { listFiles, listExamples, deleteFile, renameFile, type FileEntry } from '$lib/api';
	import { IS_STATIC_MODE, DEMO_BASE } from '$lib/static_mode';
	import { openConfirm, openPrompt } from '$lib/modals';

	let {
		active_path = $bindable<string | null>(null),
		onOpen,
		onNew,
		onOpenExample,
		onClosed,
		onSave,
		onSaveAs,
		can_save = false,
		dirty = false,
		reload = 0,
	}: {
		active_path: string | null;
		onOpen: (path: string) => void;
		onNew: () => void;
		onOpenExample?: (name: string) => void;
		onClosed?: (path: string) => void;
		onSave?: () => void;
		onSaveAs?: () => void;
		/** Whether there is an open buffer to save (enables Save / Save As). */
		can_save?: boolean;
		/** Unsaved changes in the open buffer (Save icon shows an accent dot). */
		dirty?: boolean;
		/** Bump to force a workdir re-list (e.g. after Save As creates a file). */
		reload?: number;
	} = $props();

	let files = $state<FileEntry[]>([]);
	let examples = $state<{ name: string }[]>([]);
	let workdir = $state('');
	let error = $state<string | null>(null);
	let loading = $state(false);

	// Section open state — persisted across reloads.
	const LS_SECTIONS = 'rapidfem.fb.sections';
	const LS_FOLDERS = 'rapidfem.fb.folders';

	function load_set(key: string, fallback: string[]): Set<string> {
		try {
			const raw = localStorage.getItem(key);
			if (raw) return new Set(JSON.parse(raw));
		} catch {}
		return new Set(fallback);
	}
	function save_set(key: string, s: Set<string>) {
		try { localStorage.setItem(key, JSON.stringify([...s])); } catch {}
	}

	// Sections open by default.
	let open_sections = $state<Set<string>>(load_set(LS_SECTIONS, ['workdir', 'examples']));
	// Folders default to *closed* (so a busy workdir stays manageable).
	let open_folders = $state<Set<string>>(load_set(LS_FOLDERS, ['']));

	function toggle_section(name: string) {
		if (open_sections.has(name)) open_sections.delete(name);
		else open_sections.add(name);
		open_sections = new Set(open_sections);
		save_set(LS_SECTIONS, open_sections);
	}
	function toggle_folder(path: string) {
		if (open_folders.has(path)) open_folders.delete(path);
		else open_folders.add(path);
		open_folders = new Set(open_folders);
		save_set(LS_FOLDERS, open_folders);
	}

	async function refresh() {
		loading = true;
		try {
			if (IS_STATIC_MODE) {
				// No backend: list comes from the baked manifest.json.
				workdir = '(static demo)';
				files = [];
				const r = await fetch(`${DEMO_BASE}manifest.json`);
				if (!r.ok) throw new Error(`manifest fetch failed: ${r.status}`);
				const m = await r.json();
				examples = (m.examples ?? []).map((e: { filename: string }) => ({ name: e.filename }));
			} else {
				const [r, ex] = await Promise.all([listFiles(), listExamples()]);
				workdir = r.workdir;
				files = r.files;
				examples = ex.examples;
			}
			error = null;
		} catch (e) {
			error = String(e);
		} finally {
			loading = false;
		}
	}

	$effect(() => { reload; void refresh(); });

	// ── Tree from flat paths ──────────────────────────────────────────────
	type TreeNode = {
		name: string;          // leaf name
		path: string;          // full path-from-workdir (posix-slashes)
		is_dir: boolean;
		entry?: FileEntry;     // for files
		children: TreeNode[];  // for dirs
	};

	function build_tree(entries: FileEntry[]): TreeNode {
		const root: TreeNode = { name: '', path: '', is_dir: true, children: [] };
		for (const f of entries) {
			const parts = f.path.split('/').filter(Boolean);
			let cur = root;
			for (let i = 0; i < parts.length; i++) {
				const part = parts[i];
				const is_leaf = i === parts.length - 1;
				const child_path = parts.slice(0, i + 1).join('/');
				let nxt = cur.children.find((c) => c.name === part);
				if (!nxt) {
					nxt = {
						name: part,
						path: child_path,
						is_dir: !is_leaf,
						children: [],
					};
					cur.children.push(nxt);
				}
				if (is_leaf) nxt.entry = f;
				cur = nxt;
			}
		}
		// Sort: dirs first, then files; alpha within each.
		function sort_node(n: TreeNode) {
			n.children.sort((a, b) => {
				if (a.is_dir !== b.is_dir) return a.is_dir ? -1 : 1;
				return a.name.localeCompare(b.name);
			});
			for (const c of n.children) if (c.is_dir) sort_node(c);
		}
		sort_node(root);
		return root;
	}

	let tree = $derived(build_tree(files));

	// Linearised tree honouring expansion state — easier to render with the
	// existing flat row markup and keeps row hit-targets simple.
	type Row = {
		kind: 'dir' | 'file';
		name: string;
		path: string;
		depth: number;
		open?: boolean;
		entry?: FileEntry;
	};

	function flatten(node: TreeNode, depth: number, out: Row[]) {
		for (const c of node.children) {
			if (c.is_dir) {
				const open = open_folders.has(c.path);
				out.push({ kind: 'dir', name: c.name, path: c.path, depth, open });
				if (open) flatten(c, depth + 1, out);
			} else {
				out.push({ kind: 'file', name: c.name, path: c.path, depth, entry: c.entry });
			}
		}
	}

	let rows = $derived.by(() => {
		const out: Row[] = [];
		flatten(tree, 0, out);
		return out;
	});

	// Auto-reveal the active file by opening its ancestor folders.
	$effect(() => {
		if (!active_path) return;
		const parts = active_path.split('/');
		if (parts.length <= 1) return;
		let dirty = false;
		for (let i = 1; i < parts.length; i++) {
			const p = parts.slice(0, i).join('/');
			if (!open_folders.has(p)) { open_folders.add(p); dirty = true; }
		}
		if (dirty) {
			open_folders = new Set(open_folders);
			save_set(LS_FOLDERS, open_folders);
		}
	});

	function nice_path(p: string): string {
		return p.replace(/\\/g, '/');
	}

	async function on_delete(e: MouseEvent, path: string) {
		e.stopPropagation();
		const ok = await openConfirm({
			title: 'Delete file',
			body: `This will permanently remove ${path} from your workspace.`,
			confirmLabel: 'Delete',
			danger: true,
		});
		if (!ok) return;
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
		const next = await openPrompt({
			title: 'Rename file',
			label: 'New name',
			defaultValue: path,
			confirmLabel: 'Rename',
			validate: (v) => {
				if (!v) return 'Name cannot be empty';
				if (!v.endsWith('.py')) return 'Name must end in .py';
				return null;
			},
		});
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
		<span class="title">{IS_STATIC_MODE ? 'Demo' : 'Files'}</span>
		{#if !IS_STATIC_MODE}
			<button class="tb" onclick={onNew} title="New .py file" aria-label="New file">
				<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
					<path d="M8 3v10" />
					<path d="M3 8h10" />
				</svg>
			</button>
			<button class="tb" class:dirty onclick={() => onSave?.()} disabled={!can_save} title="Save (Ctrl+S)" aria-label="Save">
				<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
					<path d="M3 2.5h7.2L13.5 5.8V13a.5.5 0 0 1-.5.5H3a.5.5 0 0 1-.5-.5V3a.5.5 0 0 1 .5-.5z" />
					<path d="M5 2.5v3.2h4.6V2.5" />
					<rect x="5" y="9" width="6" height="4.5" />
				</svg>
			</button>
			<button class="tb" onclick={() => onSaveAs?.()} disabled={!can_save} title="Save As…" aria-label="Save As">
				<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
					<path d="M3 2.5h6.2L12 5.3V8" />
					<path d="M3 2.5V13a.5.5 0 0 0 .5.5H8" />
					<path d="M5 2.5v3.2h3.6V2.5" />
					<path d="M12 10v4M10 12h4" />
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
		{/if}
	</div>

	{#if error}
		<div class="error">{error}</div>
	{/if}

	<div class="list">
		{#if !IS_STATIC_MODE}
		<!-- ── Workdir section ────────────────────────────────────────────── -->
		<div
			class="section"
			class:open={open_sections.has('workdir')}
			role="button"
			tabindex="0"
			onclick={() => toggle_section('workdir')}
			onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle_section('workdir'); } }}
			title={workdir}
		>
			<span class="chevron" class:open={open_sections.has('workdir')}>
				<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
					<polyline points="6,4 10,8 6,12" />
				</svg>
			</span>
			<span class="section-name">Workdir</span>
			<span class="section-count">{files.length}</span>
		</div>

		{#if open_sections.has('workdir')}
			{#each rows as row (row.path)}
				{#if row.kind === 'dir'}
					<div
						class="item-row dir"
						style="--depth: {row.depth}"
						role="button"
						tabindex="0"
						onclick={() => toggle_folder(row.path)}
						onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle_folder(row.path); } }}
					>
						<span class="chevron" class:open={row.open}>
							<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
								<polyline points="6,4 10,8 6,12" />
							</svg>
						</span>
						<span class="folder-icon">
							<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round">
								<path d="M2 4.5h4l1 1.5h7v6.5a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V4.5z" />
							</svg>
						</span>
						<span class="name">{row.name}</span>
					</div>
				{:else}
					<div
						class="item-row file"
						class:active={row.path === active_path}
						style="--depth: {row.depth}"
						role="button"
						tabindex="-1"
						ondblclick={(e) => on_rename(e, row.path)}
					>
						<button
							class="item"
							onclick={() => onOpen(row.path)}
							title={`${row.path}  ·  ${row.entry?.size ?? 0} bytes  ·  double-click row to rename`}
						>
							<span class="indent"></span>
							<span class="name">{row.name}</span>
						</button>
						<button class="row-act" onclick={(e) => on_delete(e, row.path)} title="Delete" aria-label="Delete">
							<svg width="11" height="11" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
								<polyline points="3,5 13,5" />
								<path d="M6 5V3h4v2" />
								<path d="M5 5l1 8h4l1-8" />
							</svg>
						</button>
					</div>
				{/if}
			{/each}
			{#if !files.length && !loading}
				<div class="empty">No .py files yet.</div>
			{/if}
		{/if}
		{/if}

		<!-- ── Examples section ───────────────────────────────────────────── -->
		{#if examples.length}
			{#if !IS_STATIC_MODE}<div class="section-sep"></div>{/if}
			<div
				class="section"
				class:open={open_sections.has('examples')}
				role="button"
				tabindex="0"
				onclick={() => toggle_section('examples')}
				onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle_section('examples'); } }}
			>
				<span class="chevron" class:open={open_sections.has('examples')}>
					<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
						<polyline points="6,4 10,8 6,12" />
					</svg>
				</span>
				<span class="section-name">Examples</span>
				<span class="section-count">{examples.length}</span>
			</div>

			{#if open_sections.has('examples')}
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
		{/if}
	</div>
</div>

<style>
	.browser {
		display: flex;
		flex-direction: column;
		height: 100%;
		background: var(--bg-mid);
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
	/* Unsaved changes: tint the Save icon with the accent colour. */
	.tb.dirty:not(:disabled) {
		color: var(--accent);
		border-color: var(--accent);
	}
	.tb svg { display: block; }

	.list { flex: 1; overflow: auto; padding: var(--space-sm) 0 var(--space-md); }

	.section {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: var(--space-sm) var(--space-lg) var(--space-xs);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		color: var(--text-dim);
		font-weight: 600;
		cursor: pointer;
		user-select: none;
		transition: color var(--transition);
	}
	.section:hover { color: var(--text-muted); }

	.chevron {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		width: 14px;
		height: 14px;
		color: var(--text-muted);
		flex-shrink: 0;
		transition: transform var(--transition), color var(--transition);
	}
	.chevron svg { display: block; }
	.chevron.open { transform: rotate(90deg); }
	.section:hover .chevron,
	.item-row.dir:hover .chevron { color: var(--text); }
	.section-name { flex: 1; }
	.section-count {
		font-size: 9px;
		color: var(--text-dim);
		font-weight: 500;
	}

	.section-sep {
		height: 1px;
		background: var(--border);
		margin: var(--space-sm) 0;
	}

	/* Tree rows */
	.item-row {
		display: flex;
		align-items: center;
		position: relative;
	}
	.item-row.dir {
		padding: 3px var(--space-lg) 3px calc(var(--space-lg) + var(--depth, 0) * 12px);
		color: var(--text-muted);
		cursor: pointer;
		user-select: none;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		gap: 4px;
	}
	.item-row.dir:hover { background: var(--bg-panel); color: var(--text); }
	.item-row.dir .folder-icon {
		display: inline-flex;
		align-items: center;
		color: var(--text-dim);
		flex-shrink: 0;
	}
	.item-row.dir .name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

	.item-row.file.active {
		background: var(--accent-dim);
		border-left: 2px solid var(--accent);
	}
	.item-row.file.active .item { color: var(--accent); }

	.item {
		display: flex;
		align-items: center;
		width: 100%;
		text-align: left;
		background: transparent;
		border: 0;
		color: var(--text-muted);
		padding: 3px var(--space-lg);
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
	.item-row.file .item {
		flex: 1;
		padding-left: calc(var(--space-lg) + var(--depth, 0) * 12px + 34px);
	}
	.item-row.file.active .item {
		padding-left: calc(var(--space-lg) + var(--depth, 0) * 12px + 34px - 2px);
	}
	.item .name {
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		display: block;
		flex: 1;
	}
	.item .indent { flex-shrink: 0; }

	.item.example { color: var(--text-dim); font-style: italic; }
	.item.example:hover { background: var(--bg-panel); color: var(--accent-secondary); }

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

	.empty {
		color: var(--text-dim);
		padding: var(--space-md) var(--space-lg);
		font-style: italic;
		font-size: var(--fs-xs);
	}
	.error { color: var(--accent); padding: var(--space-md) var(--space-lg); font-size: var(--fs-xs); }
</style>
