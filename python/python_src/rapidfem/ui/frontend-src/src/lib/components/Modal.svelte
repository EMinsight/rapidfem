<script lang="ts">
	import { modal, type ModalRequest } from '$lib/modals';
	import { onMount, tick } from 'svelte';

	// Local copy of the current request so we can read it after closing.
	// Cleared a frame later so the close animation doesn't flash empty.
	let req: ModalRequest | null = $state(null);
	let value = $state('');
	let error = $state<string | null>(null);
	let input_el: HTMLInputElement | undefined = $state();

	modal.subscribe(async (r) => {
		req = r;
		error = null;
		if (r?.kind === 'prompt') {
			value = r.defaultValue ?? '';
			await tick();
			input_el?.focus();
			input_el?.select();
		}
	});

	function cancel() {
		if (!req) return;
		// Narrow on `kind` so `resolve` resolves to the variant's own
		// signature (the union's resolve would demand a `never` argument).
		if (req.kind === 'prompt') req.resolve(null);
		else req.resolve(false);
		modal.set(null);
	}

	function confirm() {
		if (!req) return;
		if (req.kind === 'prompt') {
			const v = value.trim();
			if (req.validate) {
				const e = req.validate(v);
				if (e) {
					error = e;
					return;
				}
			}
			req.resolve(v);
		} else {
			req.resolve(true);
		}
		modal.set(null);
	}

	function on_keydown(e: KeyboardEvent) {
		if (!req) return;
		if (e.key === 'Escape') {
			e.preventDefault();
			cancel();
		} else if (e.key === 'Enter' && req.kind !== 'prompt') {
			// In prompt mode the input's own keydown handles Enter so
			// IME composition + form semantics work naturally.
			e.preventDefault();
			confirm();
		}
	}

	function on_input_keydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.isComposing) {
			e.preventDefault();
			confirm();
		}
	}

	onMount(() => {
		document.addEventListener('keydown', on_keydown);
		return () => document.removeEventListener('keydown', on_keydown);
	});
</script>

{#if req}
	<div class="backdrop" role="button" tabindex="-1" aria-label="Close dialog"
		 onclick={cancel} onkeydown={null}>
		<div class="modal" role="dialog" tabindex="-1" aria-modal="true" aria-labelledby="modal-title"
			 onclick={(e) => e.stopPropagation()} onkeydown={null}>
			<h2 id="modal-title">{req.title}</h2>

			{#if req.kind === 'confirm'}
				{#if req.body}
					<p class="body">{req.body}</p>
				{/if}
			{:else}
				{#if req.label}
					<label for="modal-input" class="label">{req.label}</label>
				{/if}
				<input id="modal-input"
					   bind:this={input_el}
					   bind:value
					   placeholder={req.placeholder ?? ''}
					   onkeydown={on_input_keydown}
					   type="text"
					   autocomplete="off"
					   spellcheck="false" />
				{#if error}
					<p class="error">{error}</p>
				{/if}
			{/if}

			<div class="actions">
				<button class="btn-cancel" onclick={cancel}>
					{req.cancelLabel ?? 'Cancel'}
				</button>
				<button class="btn-confirm" class:danger={req.kind === 'confirm' && req.danger}
						onclick={confirm}>
					{req.confirmLabel ?? (req.kind === 'prompt' ? 'OK' : 'Confirm')}
				</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.backdrop {
		position: fixed;
		inset: 0;
		background: rgba(0, 0, 0, 0.55);
		display: flex;
		align-items: center;
		justify-content: center;
		z-index: 9999;
		animation: fade-in 100ms ease-out;
	}
	@keyframes fade-in {
		from { opacity: 0; }
		to { opacity: 1; }
	}

	.modal {
		background: var(--bg-surface);
		border: 1px solid var(--border);
		min-width: 320px;
		max-width: 480px;
		padding: 18px 20px 16px;
		font-family: var(--font-mono);
		color: var(--text);
		box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
	}

	h2 {
		font-size: var(--fs-sm);
		font-weight: 700;
		color: var(--accent);
		letter-spacing: 0.5px;
		text-transform: uppercase;
		margin: 0 0 12px;
	}

	.body {
		font-size: var(--fs-xs);
		color: var(--text-muted);
		line-height: 1.5;
		margin: 0 0 16px;
		word-break: break-word;
	}

	.label {
		display: block;
		font-size: 10px;
		color: var(--text-dim);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 6px;
	}

	input {
		width: 100%;
		box-sizing: border-box;
		background: var(--bg);
		border: 1px solid var(--border);
		color: var(--text);
		font-family: var(--font-mono);
		font-size: var(--fs-xs);
		padding: 7px 10px;
		outline: none;
		transition: border-color var(--transition);
	}
	input:focus {
		border-color: var(--accent);
	}

	.error {
		font-size: 10px;
		color: var(--accent);
		margin: 6px 0 0;
		font-family: var(--font-mono);
	}

	.actions {
		display: flex;
		justify-content: flex-end;
		gap: 8px;
		margin-top: 16px;
	}

	button {
		background: transparent;
		border: 1px solid var(--border);
		color: var(--text-dim);
		font-family: var(--font-mono);
		font-size: 10px;
		font-weight: 600;
		letter-spacing: 0.5px;
		padding: 6px 14px;
		cursor: pointer;
		text-transform: uppercase;
		transition: color var(--transition), border-color var(--transition), background var(--transition);
	}
	button:hover {
		color: var(--text);
		border-color: var(--text-dim);
	}

	.btn-confirm {
		background: var(--accent);
		border-color: var(--accent);
		color: var(--bg);
	}
	.btn-confirm:hover {
		background: var(--accent-hover);
		border-color: var(--accent-hover);
		color: var(--bg);
	}
	.btn-confirm.danger {
		background: transparent;
		color: var(--accent);
	}
	.btn-confirm.danger:hover {
		background: var(--accent);
		color: var(--bg);
		border-color: var(--accent);
	}
</style>
