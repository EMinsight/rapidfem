/**
 * Main-thread client for the splat depth-sort worker.
 *
 * Owns the Worker, hands it the splat positions once per cloud, and debounces
 * sort requests: only the latest view direction matters, so an in-flight sort
 * is allowed to finish and the newest pending request fires right after. A
 * monotonic token drops stale results.
 *
 * Usage:
 *   const sorter = new SplatSorter((index) => { setSplatOrder(gl, index); render(); });
 *   sorter.load(positions);                 // once per resample
 *   sorter.requestSort([vx, vy, vz]);        // on every camera change
 *   sorter.dispose();                        // on teardown
 */

export class SplatSorter {
	private worker: Worker;
	private token = 0;
	private lastApplied = -1;
	private inFlight = false;
	private pendingView: [number, number, number] | null = null;
	private onSorted: (index: Uint32Array) => void;

	constructor(onSorted: (index: Uint32Array) => void) {
		this.onSorted = onSorted;
		this.worker = new Worker(new URL('./splat_sort_worker.ts', import.meta.url), {
			type: 'module',
		});
		this.worker.onmessage = (e: MessageEvent) => {
			const { type, index, token } = e.data as {
				type: string;
				index: Uint32Array;
				token: number;
			};
			if (type !== 'sorted') return;
			this.inFlight = false;
			// Apply only if newer than what's on the GPU (out-of-order guard).
			if (token > this.lastApplied) {
				this.lastApplied = token;
				this.onSorted(index);
			}
			// A view change arrived while we were busy — service it now.
			if (this.pendingView) {
				const v = this.pendingView;
				this.pendingView = null;
				this.dispatch(v);
			}
		};
	}

	/** Hand the worker a fresh set of splat positions. Copies, then transfers
	 *  the copy — the caller keeps its own array intact. */
	load(positions: Float32Array): void {
		const copy = positions.slice();
		this.worker.postMessage({ type: 'load', positions: copy }, [copy.buffer]);
		// Existing draw order is stale; force the next sort to apply.
		this.lastApplied = -1;
	}

	/** Request a sort for the given camera view direction (into the scene). */
	requestSort(view: [number, number, number]): void {
		if (this.inFlight) {
			this.pendingView = view;
			return;
		}
		this.dispatch(view);
	}

	private dispatch(view: [number, number, number]): void {
		this.inFlight = true;
		this.worker.postMessage({ type: 'sort', view, token: ++this.token });
	}

	dispose(): void {
		this.worker.terminate();
	}
}
