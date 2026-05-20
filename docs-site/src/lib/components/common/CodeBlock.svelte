<script lang="ts">
	import Icon from './Icon.svelte';
	import { tooltip } from './Tooltip.svelte';

	interface Props {
		/** The code to display */
		code: string;
		/** Title shown in the panel header */
		title?: string;
		/** Language hint (cosmetic, shown as a label) */
		lang?: string;
	}

	let { code, title = 'Code', lang }: Props = $props();

	let copied = $state(false);

	async function handleCopy() {
		try {
			await navigator.clipboard.writeText(code);
			copied = true;
			setTimeout(() => (copied = false), 2000);
		} catch {
			copied = false;
		}
	}
</script>

<div class="code-panel">
	<div class="panel-header">
		<span>{title}{#if lang}<span class="lang">{lang}</span>{/if}</span>
		<button
			class="icon-btn"
			class:copied
			onclick={handleCopy}
			use:tooltip={copied ? 'Copied!' : 'Copy'}
		>
			<Icon name={copied ? 'check' : 'copy'} size={14} />
		</button>
	</div>
	<div class="panel-body">
		<pre><code>{code}</code></pre>
	</div>
</div>

<style>
	.lang {
		margin-left: var(--space-sm);
		color: var(--text-disabled);
		font-weight: 500;
	}

	.panel-body pre {
		color: var(--text);
	}
</style>
