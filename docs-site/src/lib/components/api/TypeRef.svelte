<script lang="ts">
	// Renders a type annotation string in monospace. Fully-qualified paths
	// are collapsed to their final segment for readability.
	interface Props {
		type: string;
	}

	let { type }: Props = $props();

	let display = $derived(
		type
			.split(/(\[|\]|,\s*|\s*\|\s*)/)
			.map((token) => {
				if (/^[a-z][\w.]*\.[A-Z]\w*$/.test(token)) {
					return token.split('.').pop() ?? token;
				}
				return token;
			})
			.join('')
	);
</script>

<span class="type-ref">{display}</span>

<style>
	.type-ref {
		font-family: var(--font-mono);
		color: var(--text-muted);
	}
</style>
