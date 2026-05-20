<script lang="ts" module>
	import { writable } from 'svelte/store';

	interface TooltipState {
		text: string;
		shortcut?: string;
		maxWidth?: number;
		x: number;
		y: number;
		visible: boolean;
		position: 'bottom' | 'left' | 'right' | 'top';
	}

	export const tooltipStore = writable<TooltipState>({
		text: '',
		x: 0,
		y: 0,
		visible: false,
		position: 'bottom'
	});

	let showTimeout: ReturnType<typeof setTimeout> | null = null;
	let hideTimeout: ReturnType<typeof setTimeout> | null = null;

	export function showTooltip(
		text: string,
		element: HTMLElement,
		position: 'bottom' | 'left' | 'right' | 'top' = 'bottom',
		shortcut?: string,
		maxWidth?: number
	) {
		if (hideTimeout) {
			clearTimeout(hideTimeout);
			hideTimeout = null;
		}

		showTimeout = setTimeout(() => {
			const rect = element.getBoundingClientRect();
			let x: number, y: number;

			switch (position) {
				case 'left':
					x = rect.left - 8;
					y = rect.top + rect.height / 2;
					break;
				case 'right':
					x = rect.right + 8;
					y = rect.top + rect.height / 2;
					break;
				case 'top':
					x = rect.left + rect.width / 2;
					y = rect.top - 8;
					break;
				case 'bottom':
				default:
					x = rect.left + rect.width / 2;
					y = rect.bottom + 8;
					break;
			}

			tooltipStore.set({ text, shortcut, maxWidth, x, y, visible: true, position });
		}, 50);
	}

	export function hideTooltip() {
		if (showTimeout) {
			clearTimeout(showTimeout);
			showTimeout = null;
		}
		hideTimeout = setTimeout(() => {
			tooltipStore.update((s) => ({ ...s, visible: false }));
		}, 50);
	}

	type TooltipParams = string | { text: string; shortcut?: string; maxWidth?: number; position?: 'bottom' | 'left' | 'right' | 'top' };

	// Svelte action for easy tooltip usage
	export function tooltip(node: HTMLElement, params: TooltipParams) {
		let text = typeof params === 'string' ? params : params.text;
		let shortcut = typeof params === 'string' ? undefined : params.shortcut;
		let maxWidth = typeof params === 'string' ? undefined : params.maxWidth;
		let position = typeof params === 'string' ? 'bottom' : (params.position ?? 'bottom');

		function handleMouseEnter() {
			if (text) showTooltip(text, node, position, shortcut, maxWidth);
		}

		function handleMouseLeave() {
			hideTooltip();
		}

		node.addEventListener('mouseenter', handleMouseEnter);
		node.addEventListener('mouseleave', handleMouseLeave);

		return {
			update(newParams: TooltipParams) {
				// Just update the captured variables - handlers reference them via closure
				text = typeof newParams === 'string' ? newParams : newParams.text;
				shortcut = typeof newParams === 'string' ? undefined : newParams.shortcut;
				maxWidth = typeof newParams === 'string' ? undefined : newParams.maxWidth;
				position = typeof newParams === 'string' ? 'bottom' : (newParams.position ?? 'bottom');
			},
			destroy() {
				node.removeEventListener('mouseenter', handleMouseEnter);
				node.removeEventListener('mouseleave', handleMouseLeave);
				hideTooltip();
			}
		};
	}
</script>

<script lang="ts">
	import { tick } from 'svelte';

	let tooltipState: TooltipState = $state({ text: '', x: 0, y: 0, visible: false, position: 'bottom' });
	let tooltipEl: HTMLDivElement | undefined = $state();
	let adjustment: { x: number; y: number } = $state({ x: 0, y: 0 });

	tooltipStore.subscribe(async (s) => {
		tooltipState = s;
		adjustment = { x: 0, y: 0 }; // Reset adjustment
		if (s.visible) {
			// Wait for DOM update, then check viewport collision
			await tick();
			adjustPosition();
		}
	});

	function adjustPosition() {
		if (!tooltipEl) return;

		const rect = tooltipEl.getBoundingClientRect();
		const padding = 8;
		let adjustX = 0;
		let adjustY = 0;

		// Check horizontal overflow
		if (rect.left < padding) {
			adjustX = padding - rect.left;
		} else if (rect.right > window.innerWidth - padding) {
			adjustX = window.innerWidth - padding - rect.right;
		}

		// Check vertical overflow
		if (rect.top < padding) {
			adjustY = padding - rect.top;
		} else if (rect.bottom > window.innerHeight - padding) {
			adjustY = window.innerHeight - padding - rect.bottom;
		}

		adjustment = { x: adjustX, y: adjustY };
	}

	// Compute full transform based on position and adjustment
	let transform = $derived.by(() => {
		const { x, y } = adjustment;
		const adj = (x !== 0 || y !== 0) ? ` translate(${x}px, ${y}px)` : '';
		switch (tooltipState.position) {
			case 'top':
				return `translateX(-50%) translateY(-100%)${adj}`;
			case 'left':
				return `translateX(-100%) translateY(-50%)${adj}`;
			case 'right':
				return `translateY(-50%)${adj}`;
			case 'bottom':
			default:
				return `translateX(-50%)${adj}`;
		}
	});
</script>

{#if tooltipState.visible}
	<div
		bind:this={tooltipEl}
		class="tooltip"
		style="left: {tooltipState.x}px; top: {tooltipState.y}px; transform: {transform};{tooltipState.maxWidth ? ` max-width: ${tooltipState.maxWidth}px;` : ''}"
	>
		<span class="text">{tooltipState.text}</span>
		{#if tooltipState.shortcut}
			<span class="shortcut">{tooltipState.shortcut}</span>
		{/if}
	</div>
{/if}

<style>
	.tooltip {
		position: fixed;
		display: flex;
		align-items: baseline;
		gap: 8px;
		padding: 4px 8px;
		max-width: 240px;
		background: var(--surface);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		font-size: var(--font-base);
		color: var(--text-muted);
		pointer-events: none;
		z-index: 10000;
		box-shadow: var(--shadow-sm);
		animation: fadeIn var(--transition-fast) ease-out;
	}

	.shortcut {
		font-family: var(--font-mono);
		font-size: var(--font-base);
		color: var(--text-disabled);
		white-space: nowrap;
	}

	@keyframes fadeIn {
		from { opacity: 0; }
		to { opacity: 1; }
	}
</style>
