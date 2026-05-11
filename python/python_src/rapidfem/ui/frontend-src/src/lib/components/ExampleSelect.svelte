<script lang="ts">
	import '$lib/components/fields.css';
	import { EXAMPLES, type DemoExample } from '$lib/examples';

	let { value = $bindable<string>() }: { value: string } = $props();
	let open = $state(false);

	function pick(id: string) {
		value = id;
		open = false;
	}

	const items: DemoExample[] = Object.values(EXAMPLES);
</script>

<svelte:window onclick={() => (open = false)} />

<div class="ex-dropdown" onclick={(e) => e.stopPropagation()} role="presentation">
	<button class="ex-btn" onclick={() => (open = !open)}>
		{EXAMPLES[value]?.label ?? value}
		<svg width="8" height="5" viewBox="0 0 8 5" fill="currentColor"><path d="M0 0L4 5L8 0Z" /></svg>
	</button>
	{#if open}
		<div class="ex-menu">
			{#each items as ex}
				<button class="ex-option" class:active={value === ex.id} onclick={() => pick(ex.id)}>
					<span class="ex-name">{ex.label}</span>
					<span class="ex-desc">{ex.description}</span>
				</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.ex-dropdown {
		position: relative;
	}
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
		text-transform: none;
		letter-spacing: 0;
		font-weight: 500;
		transition: border-color var(--transition);
	}
	.ex-btn:hover {
		border-color: var(--accent);
	}
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
		display: flex;
		flex-direction: column;
		gap: 1px;
		transition: background var(--transition);
	}
	.ex-option:hover {
		background: var(--accent-dim);
	}
	.ex-option.active .ex-name {
		color: var(--accent);
		font-weight: 600;
	}
	.ex-name {
		font-weight: 500;
	}
	.ex-desc {
		font-size: 9px;
		color: var(--text-dim);
		line-height: 1.3;
	}
</style>
