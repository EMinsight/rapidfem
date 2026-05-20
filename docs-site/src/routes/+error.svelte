<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/stores';
</script>

<div class="error-page">
	<div class="error-code">{$page.status}</div>
	<p class="error-message">{$page.error?.message ?? 'Page not found'}</p>
	<a href="{base}/" class="error-link">Back to documentation</a>
</div>

<style>
	.error-page {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: var(--space-md);
		padding: var(--space-3xl);
		text-align: center;
	}

	.error-code {
		font-family: var(--font-mono);
		font-size: 64px;
		font-weight: 600;
		color: var(--accent);
		line-height: 1;
	}

	.error-message {
		color: var(--text-muted);
		margin: 0;
	}

	.error-link {
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-weight: 600;
		font-size: var(--fs-sm);
	}
</style>
