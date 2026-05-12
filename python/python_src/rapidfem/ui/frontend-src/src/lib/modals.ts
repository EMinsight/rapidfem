/**
 * Promise-based modals that match the rest of the rapidfem UI styling.
 *
 * Replaces native ``window.confirm()`` and ``window.prompt()`` (which look
 * out of place against the dark mono-font UI). The Modal component lives
 * once in ``+layout.svelte`` and reads from this store; any component can
 * pop a modal with::
 *
 *     const ok = await openConfirm({ title: 'Delete?', body: '...' });
 *     const name = await openPrompt({ title: 'Rename', defaultValue: 'x.py' });
 */
import { writable } from 'svelte/store';

interface ConfirmRequest {
	kind: 'confirm';
	title: string;
	body?: string;
	confirmLabel?: string;
	cancelLabel?: string;
	danger?: boolean;
	resolve: (ok: boolean) => void;
}

interface PromptRequest {
	kind: 'prompt';
	title: string;
	label?: string;
	placeholder?: string;
	defaultValue?: string;
	confirmLabel?: string;
	cancelLabel?: string;
	validate?: (v: string) => string | null;  // return error message or null
	resolve: (value: string | null) => void;
}

export type ModalRequest = ConfirmRequest | PromptRequest;

export const modal = writable<ModalRequest | null>(null);

export function openConfirm(opts: Omit<ConfirmRequest, 'kind' | 'resolve'>): Promise<boolean> {
	return new Promise((resolve) => {
		modal.set({ kind: 'confirm', resolve, ...opts });
	});
}

export function openPrompt(opts: Omit<PromptRequest, 'kind' | 'resolve'>): Promise<string | null> {
	return new Promise((resolve) => {
		modal.set({ kind: 'prompt', resolve, ...opts });
	});
}
