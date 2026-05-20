<script lang="ts">
	import Icon from './Icon.svelte';
	import CodeMirror from './CodeMirror.svelte';
	import { tooltip } from './Tooltip.svelte';

	interface Props {
		/** The code to display. */
		code: string;
		/** Title shown in the panel header. */
		title?: string;
		/** Language hint, shown as a label. */
		lang?: string;
		/** Show line numbers. */
		lineNumbers?: boolean;
	}

	let { code, title = 'Code', lang, lineNumbers = true }: Props = $props();

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
	<CodeMirror {code} {lineNumbers} />
</div>

<style>
	.lang {
		margin-left: var(--space-sm);
		color: var(--text-disabled);
		font-weight: 500;
	}
</style>
