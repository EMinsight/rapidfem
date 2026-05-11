<script lang="ts">
	let {
		onDelta,
		onStart,
		onEnd,
		vertical = false,
	}: {
		onDelta: (d: number) => void;
		onStart?: () => void;
		onEnd?: () => void;
		vertical?: boolean;
	} = $props();

	let dragging = $state(false);
	let last = 0;

	function on_down(e: PointerEvent) {
		dragging = true;
		last = vertical ? e.clientY : e.clientX;
		(e.target as HTMLElement).setPointerCapture(e.pointerId);
		document.body.style.cursor = vertical ? 'row-resize' : 'col-resize';
		document.body.style.userSelect = 'none';
		onStart?.();
	}

	function on_move(e: PointerEvent) {
		if (!dragging) return;
		const cur = vertical ? e.clientY : e.clientX;
		const d = cur - last;
		last = cur;
		if (d !== 0) onDelta(d);
	}

	function on_up(e: PointerEvent) {
		if (!dragging) return;
		dragging = false;
		(e.target as HTMLElement).releasePointerCapture(e.pointerId);
		document.body.style.cursor = '';
		document.body.style.userSelect = '';
		onEnd?.();
	}
</script>

<div
	class="resizer"
	class:vertical
	class:dragging
	role="separator"
	aria-orientation={vertical ? 'horizontal' : 'vertical'}
	tabindex="-1"
	onpointerdown={on_down}
	onpointermove={on_move}
	onpointerup={on_up}
	onpointercancel={on_up}
></div>

<style>
	.resizer {
		flex: 0 0 4px;
		background: var(--border);
		cursor: col-resize;
		transition: background var(--transition);
		touch-action: none;
		z-index: 5;
	}
	.resizer.vertical {
		cursor: row-resize;
	}
	.resizer:hover,
	.resizer.dragging {
		background: var(--accent);
	}
</style>
