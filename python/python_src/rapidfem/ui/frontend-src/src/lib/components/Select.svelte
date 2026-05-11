<script lang="ts">
	import '$lib/components/fields.css';

	let {
		value = $bindable<string | number>(),
		options
	}: {
		value: string | number;
		options: { value: string | number; label: string }[];
	} = $props();
	let open = $state(false);

	const current_label = $derived(
		options.find((o) => o.value === value)?.label ?? String(value)
	);
</script>

<svelte:window onclick={() => (open = false)} />

<div class="ex-dropdown" onclick={(e) => e.stopPropagation()} role="presentation">
	<button class="ex-btn" onclick={() => (open = !open)}>
		{current_label}
		<svg width="8" height="5" viewBox="0 0 8 5" fill="currentColor"><path d="M0 0L4 5L8 0Z" /></svg>
	</button>
	{#if open}
		<div class="ex-menu">
			{#each options as o}
				<button
					class="ex-option"
					class:active={value === o.value}
					onclick={() => { value = o.value; open = false; }}
				>
					{o.label}
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.ex-dropdown { position: relative; }
	.ex-btn {
		width: 100%;
		padding: 5px 8px;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		background: var(--input-bg);
		border: 1px solid var(--input-border);
		color: var(--text-muted);
		cursor: pointer;
		text-align: left;
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 8px;
		text-transform: none;
		letter-spacing: 0;
		font-weight: 500;
		transition: border-color var(--transition);
	}
	.ex-btn:hover { border-color: var(--accent); color: var(--text); }
	.ex-menu {
		position: absolute;
		top: 100%;
		left: 0;
		right: 0;
		z-index: 20;
		background: var(--bg-surface);
		border: 1px solid var(--border);
		display: flex;
		flex-direction: column;
	}
	.ex-option {
		padding: 6px 8px;
		font-size: var(--fs-xs);
		font-family: var(--font-mono);
		color: var(--text-muted);
		background: none;
		border: none;
		text-align: left;
		cursor: pointer;
		text-transform: none;
		letter-spacing: 0;
		font-weight: 400;
		transition: background var(--transition);
	}
	.ex-option:hover { background: var(--accent-dim); color: var(--text); }
	.ex-option.active { color: var(--accent); font-weight: 600; }
</style>
