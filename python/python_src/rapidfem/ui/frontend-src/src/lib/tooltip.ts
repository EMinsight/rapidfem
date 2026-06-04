/** Global smart tooltips.
 *
 * Markup stays declarative: any element with class `has-tip` holds a child
 * `<span class="tip">…</span>` (optionally `.up` to prefer placement above).
 * This controller renders the tip into a single body-level element positioned
 * with `position: fixed`, so it never clips against an `overflow` ancestor
 * (e.g. the file list) and is flipped / clamped to stay inside the viewport.
 *
 * The original `.tip` spans are kept hidden in the DOM purely as the content
 * source, so every call site keeps working without change.
 */

const MARGIN = 6; // gap between trigger and tip
const EDGE = 4; // min distance from the viewport edge

let box: HTMLDivElement | null = null;
let current: Element | null = null;

function ensureBox(): HTMLDivElement {
	if (box) return box;
	box = document.createElement('div');
	box.className = 'smart-tip';
	box.setAttribute('role', 'tooltip');
	document.body.appendChild(box);
	return box;
}

function tipOf(trigger: Element): HTMLElement | null {
	return trigger.querySelector(':scope > .tip');
}

function show(trigger: Element, tip: HTMLElement): void {
	const el = ensureBox();
	el.innerHTML = tip.innerHTML;
	el.style.visibility = 'hidden';
	el.style.display = 'flex';

	const tr = trigger.getBoundingClientRect();
	const tw = el.offsetWidth;
	const th = el.offsetHeight;
	const vw = window.innerWidth;
	const vh = window.innerHeight;

	// Vertical: prefer below, unless `.up` is hinted or there is no room below.
	const wantAbove = tip.classList.contains('up');
	const roomBelow = vh - tr.bottom - MARGIN >= th + EDGE;
	const roomAbove = tr.top - MARGIN >= th + EDGE;
	let top: number;
	if ((wantAbove && roomAbove) || (!roomBelow && roomAbove)) {
		top = tr.top - MARGIN - th;
	} else {
		top = tr.bottom + MARGIN;
	}

	// Horizontal: centred on the trigger, then clamped into the viewport.
	let left = tr.left + tr.width / 2 - tw / 2;
	left = Math.max(EDGE, Math.min(left, vw - tw - EDGE));
	top = Math.max(EDGE, Math.min(top, vh - th - EDGE));

	el.style.left = `${Math.round(left)}px`;
	el.style.top = `${Math.round(top)}px`;
	el.style.visibility = 'visible';
	current = trigger;
}

function hide(): void {
	if (box) box.style.display = 'none';
	current = null;
}

/** Wire up the document-level listeners. Returns a teardown function. */
export function initTooltips(): () => void {
	if (typeof document === 'undefined') return () => {};

	const onOver = (e: Event) => {
		const target = e.target as Element | null;
		const trigger = target?.closest?.('.has-tip');
		if (!trigger || trigger === current) return;
		if ((trigger as HTMLButtonElement).disabled) return; // no tip on disabled
		const tip = tipOf(trigger);
		if (!tip) return;
		show(trigger, tip);
	};
	const onOut = (e: MouseEvent) => {
		const target = e.target as Element | null;
		const trigger = target?.closest?.('.has-tip');
		if (!trigger || trigger !== current) return;
		// Ignore moves that stay within the same trigger.
		const to = e.relatedTarget as Node | null;
		if (to && trigger.contains(to)) return;
		hide();
	};

	document.addEventListener('mouseover', onOver, true);
	document.addEventListener('mouseout', onOut, true);
	document.addEventListener('focusin', onOver, true);
	document.addEventListener('focusout', () => hide(), true);
	// Positions go stale on scroll/resize; just dismiss.
	window.addEventListener('scroll', hide, true);
	window.addEventListener('resize', hide);

	return () => {
		document.removeEventListener('mouseover', onOver, true);
		document.removeEventListener('mouseout', onOut, true);
		document.removeEventListener('focusin', onOver, true);
		window.removeEventListener('scroll', hide, true);
		window.removeEventListener('resize', hide);
		if (box) { box.remove(); box = null; }
		current = null;
	};
}
