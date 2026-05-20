<script lang="ts">
	import { base } from '$app/paths';
	import Icon from '$lib/components/common/Icon.svelte';
	import { tooltip } from '$lib/components/common/Tooltip.svelte';
	import { external } from '$lib/config/rapidfem';

	interface Props {
		onMenuClick?: () => void;
	}

	let { onMenuClick }: Props = $props();
</script>

<header>
	<div class="header-content">
		<div class="header-left">
			{#if onMenuClick}
				<button class="icon-btn menu-btn" onclick={onMenuClick} use:tooltip={'Menu'}>
					<Icon name="menu" size={14} />
				</button>
			{/if}
			<a href="{base}/" class="wordmark">
				rapid<span class="wordmark-accent">fem</span>
				<span class="wordmark-tag">docs</span>
			</a>
		</div>
		<div class="header-right">
			<a href={external.demo} class="icon-btn" use:tooltip={'Examples / Demo'}>
				<Icon name="play" size={14} />
			</a>
			<a href={external.pypi} class="icon-btn" use:tooltip={'PyPI'}>
				<Icon name="package" size={14} />
			</a>
			<a href={external.github} class="icon-btn" use:tooltip={'GitHub'}>
				<Icon name="github" size={14} />
			</a>
		</div>
	</div>
</header>

<style>
	header {
		flex-shrink: 0;
		height: var(--header-height);
		background: var(--surface-raised);
		border-bottom: 1px solid var(--border);
	}

	.header-content {
		height: 100%;
		padding: 0 var(--space-lg);
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.header-left {
		display: flex;
		align-items: center;
		gap: var(--space-md);
	}

	.header-right {
		display: flex;
		align-items: center;
		gap: var(--space-xs);
	}

	.menu-btn {
		display: none;
	}

	@media (max-width: 768px) {
		.menu-btn {
			display: flex;
		}
	}

	.wordmark {
		display: flex;
		align-items: baseline;
		gap: 6px;
		font-family: var(--font-mono);
		font-size: var(--font-md);
		font-weight: 600;
		letter-spacing: -0.02em;
		color: var(--text);
		text-decoration: none;
		text-transform: lowercase;
	}

	.wordmark:hover {
		text-decoration: none;
	}

	.wordmark-accent {
		color: var(--accent);
	}

	.wordmark-tag {
		font-size: var(--font-base);
		font-weight: 500;
		color: var(--text-disabled);
		text-transform: uppercase;
		letter-spacing: 1px;
	}
</style>
