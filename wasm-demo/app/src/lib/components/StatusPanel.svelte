<script lang="ts">
	import '$lib/components/fields.css';

	let {
		running = false,
		status = 'idle',
		progress = 0,
		log_lines = [],
		onrun,
		onabort
	}: {
		running?: boolean;
		status?: string;
		progress?: number;
		log_lines?: string[];
		onrun: () => void;
		onabort: () => void;
	} = $props();
</script>

<div class="param-section status-section">
	<h4>Run</h4>
	<div class="actions">
		{#if !running}
			<button class="primary" onclick={onrun}>Run sweep</button>
		{:else}
			<button class="secondary" onclick={onabort}>Abort</button>
		{/if}
	</div>
	<div class="status-row">
		<span class="status">{status}</span>
		<span class="progress-pct">{Math.round(progress * 100)}%</span>
	</div>
	<div class="progress-bar">
		<div class="progress-fill" style="width: {progress * 100}%"></div>
	</div>
</div>

<div class="param-section log-section">
	<h4>Log</h4>
	<pre class="log">{log_lines.join('\n') || '—'}</pre>
</div>

<style>
	.status-section {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.actions { display: flex; gap: 4px; }
	button {
		flex: 1;
		padding: 8px 12px;
		font-family: var(--font-mono);
		font-size: var(--fs-sm);
		text-transform: none;
		letter-spacing: 0.5px;
		border: 1px solid var(--input-border);
		cursor: pointer;
		transition: background var(--transition), color var(--transition);
	}
	button.primary {
		background: var(--accent);
		color: var(--bg);
		border-color: var(--accent);
	}
	button.primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
	button.secondary {
		background: var(--bg-panel);
		color: var(--text-muted);
	}
	button.secondary:hover {
		background: var(--accent);
		color: var(--bg);
		border-color: var(--accent);
	}
	.status-row {
		display: flex;
		justify-content: space-between;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
	}
	.progress-pct { color: var(--accent); }
	.progress-bar {
		height: 3px;
		background: var(--bg-inset);
		overflow: hidden;
	}
	.progress-fill {
		height: 100%;
		background: var(--accent);
		transition: width 0.15s ease-out;
	}
	.log-section {
		flex: 1;
		min-height: 0;
		display: flex;
		flex-direction: column;
	}
	.log {
		flex: 1;
		background: var(--bg-inset);
		border: 1px solid var(--border-subtle);
		padding: 6px 8px;
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		color: var(--text-muted);
		overflow-y: auto;
		max-height: 240px;
		white-space: pre-wrap;
		margin: 0;
		line-height: 1.5;
	}
</style>
