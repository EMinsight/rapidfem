/**
 * WS-based kernel client. Single ordered event stream per cell execution.
 *
 * Protocol (mirrors rapidfem.ui.kernel_ws):
 *   client → execute / reset / interrupt
 *   server → hello / started / stream / display / error / done
 */

import type { GeometryPayload, MeshPayload, PythonError } from './api';

export type StreamKind = 'stdout' | 'stderr';

export type KernelEvent =
	| { type: 'hello' }
	| { type: 'started'; cell_id: string; file: string }
	| { type: 'stream'; cell_id: string; stream: StreamKind; line: string }
	| { type: 'display'; cell_id: string; kind: 'geometry'; name: string; payload: GeometryPayload }
	| { type: 'display'; cell_id: string; kind: 'mesh'; name: string; payload: MeshPayload }
	| { type: 'display'; cell_id: string; kind: 'result'; name: string; payload: SolveResultPayload }
	| { type: 'display'; cell_id: string; kind: 'error'; name: string; error: PythonError }
	| { type: 'error'; cell_id: string; error: PythonError }
	| { type: 'done'; cell_id: string; ok: boolean }
	| { type: 'reset_ack'; file: string }
	| { type: 'interrupt_ack'; cell_id: string; ok: boolean };

export interface SolveResultPayload {
	frequencies: number[];
	sparams: number[][][][];
	n_driven: number;
	n_freq: number;
	n_dofs: number;
	n_tets: number;
	solve_time_s: number;
	fields?: (number[] | null)[][];
}

export interface ExecuteOptions {
	cell_id: string;
	file: string;
	code: string;
	reset?: boolean;
	onStarted?: () => void;
	onStream?: (stream: StreamKind, line: string) => void;
	onDisplay?: (kind: 'geometry' | 'mesh' | 'result', payload: unknown, name: string) => void;
}

export interface ExecuteResult {
	ok: boolean;
	error?: PythonError;
}

function ws_url(): string {
	if (typeof window === 'undefined') return '';
	const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
	if (window.location.port === '5173') return `${proto}//127.0.0.1:5174/ws/kernel`;
	return `${proto}//${window.location.host}/ws/kernel`;
}

export class KernelClient {
	private ws: WebSocket | null = null;
	private pending = new Map<string, {
		opts: ExecuteOptions;
		resolve: (r: ExecuteResult) => void;
	}>();
	private outbox: string[] = [];
	private connect_retry: ReturnType<typeof setTimeout> | null = null;
	private hello_resolvers: Array<() => void> = [];
	private connected = false;

	constructor() {
		if (typeof window !== 'undefined') this.connect();
	}

	private connect() {
		try {
			this.ws = new WebSocket(ws_url());
		} catch {
			this.schedule_reconnect();
			return;
		}
		this.ws.onmessage = (m) => {
			try {
				this.dispatch(JSON.parse(m.data) as KernelEvent);
			} catch (err) {
				console.warn('[kernel] bad payload', err);
			}
		};
		this.ws.onopen = () => {
			this.connected = true;
			for (const msg of this.outbox) this.ws?.send(msg);
			this.outbox = [];
		};
		this.ws.onerror = () => {};
		this.ws.onclose = () => {
			this.connected = false;
			this.ws = null;
			// Fail any in-flight cells so callers don't hang.
			for (const [_id, p] of this.pending) {
				p.resolve({ ok: false, error: { type: 'ConnectionError', message: 'WS closed', traceback: '' } });
			}
			this.pending.clear();
			this.schedule_reconnect();
		};
	}

	private schedule_reconnect() {
		if (this.connect_retry) return;
		this.connect_retry = setTimeout(() => {
			this.connect_retry = null;
			this.connect();
		}, 1500);
	}

	private send_raw(msg: object) {
		const s = JSON.stringify(msg);
		if (this.ws && this.connected) this.ws.send(s);
		else this.outbox.push(s);
	}

	private dispatch(e: KernelEvent) {
		if (e.type === 'hello') {
			const rs = this.hello_resolvers;
			this.hello_resolvers = [];
			for (const r of rs) r();
			return;
		}
		const cell_id = (e as { cell_id?: string }).cell_id;
		if (!cell_id) return;
		const p = this.pending.get(cell_id);
		if (!p) return;
		if (e.type === 'started') {
			p.opts.onStarted?.();
		} else if (e.type === 'stream') {
			p.opts.onStream?.(e.stream, e.line);
		} else if (e.type === 'display') {
			if (e.kind === 'error') {
				// Surface as stream so the user sees it in the output panel.
				p.opts.onStream?.('stderr', `${e.error.type}: ${e.error.message}`);
			} else {
				p.opts.onDisplay?.(e.kind, e.payload, e.name);
			}
		} else if (e.type === 'error') {
			p.opts.onStream?.('stderr', `${e.error.type}: ${e.error.message}`);
			// keep pending; "done" closes the promise with ok:false
		} else if (e.type === 'done') {
			this.pending.delete(cell_id);
			p.resolve({ ok: e.ok });
		}
	}

	execute(opts: ExecuteOptions): Promise<ExecuteResult> {
		return new Promise((resolve) => {
			this.pending.set(opts.cell_id, { opts, resolve });
			this.send_raw({
				type: 'execute',
				cell_id: opts.cell_id,
				file: opts.file,
				code: opts.code,
				reset: !!opts.reset,
			});
		});
	}

	reset(file: string) {
		this.send_raw({ type: 'reset', file });
	}
}

// ── Static-demo replay client ──────────────────────────────────────────
//
// Used when the frontend is built with VITE_STATIC_MODE=1. Loads the bake
// artefacts produced by `scripts/bake_demo.py` from `static/demo/` and
// replays them through the same execute() callbacks as the live WS path,
// so consumers (Cell, Notebook, +page.svelte) don't have to branch.

import { IS_STATIC_MODE, DEMO_BASE } from './static_mode';

interface BakedFieldsStub {
	$bin: true;
	magic: number;
	version: number;
	n_freq: number;
	n_port: number;
	stride: number;
	url: string;
}

interface BakedDisplayEvent {
	kind: 'geometry' | 'mesh' | 'result' | 'error';
	name: string;
	payload?: Record<string, unknown>;
	error?: PythonError;
}

interface BakedCell {
	marker: string | null;
	code: string;
	status: 'ok' | 'error';
	stream_lines: { stream: StreamKind; line: string }[];
	display_events: BakedDisplayEvent[];
	error?: PythonError;
}

interface BakedExample {
	name: string;
	filename: string;
	source: string;
	cells: BakedCell[];
}

interface ManifestEntry {
	name: string;
	filename: string;
	json: string;
	bin_files: string[];
	n_cells: number;
}

interface Manifest {
	version: number;
	baked_at: number;
	examples: ManifestEntry[];
}

class StaticKernelClient {
	private manifest_promise: Promise<Manifest> | null = null;
	private examples = new Map<string, Promise<BakedExample>>();
	private bins = new Map<string, Promise<ArrayBuffer>>();

	private load_manifest(): Promise<Manifest> {
		if (!this.manifest_promise) {
			this.manifest_promise = fetch(`${DEMO_BASE}manifest.json`)
				.then((r) => {
					if (!r.ok) throw new Error(`manifest fetch failed: ${r.status}`);
					return r.json() as Promise<Manifest>;
				});
		}
		return this.manifest_promise;
	}

	private async load_example(filename: string): Promise<BakedExample> {
		// File path keys are stored as bare filenames in the manifest, but
		// callers may pass a longer "<path>" — match on basename.
		const base = filename.split(/[\\/]/).pop() ?? filename;
		let p = this.examples.get(base);
		if (p) return p;
		p = this.load_manifest().then(async (m) => {
			const entry = m.examples.find((e) => e.filename === base);
			if (!entry) throw new Error(`baked example not found: ${base}`);
			const r = await fetch(`${DEMO_BASE}${entry.json}`);
			if (!r.ok) throw new Error(`example fetch failed: ${r.status}`);
			return r.json() as Promise<BakedExample>;
		});
		this.examples.set(base, p);
		return p;
	}

	private async load_bin(url: string): Promise<ArrayBuffer> {
		let p = this.bins.get(url);
		if (p) return p;
		p = fetch(`${DEMO_BASE}${url}`).then((r) => {
			if (!r.ok) throw new Error(`bin fetch failed: ${r.status} ${url}`);
			return r.arrayBuffer();
		});
		this.bins.set(url, p);
		return p;
	}

	/** Hydrate the binary field stub back into the nested-array shape the
	 *  live UI's `fields_raw` expects: ``(number[] | null)[][]``. */
	private async hydrate_fields(stub: BakedFieldsStub): Promise<(number[] | null)[][]> {
		const buf = await this.load_bin(stub.url);
		const dv = new DataView(buf);
		const magic = dv.getUint32(0, true);
		const version = dv.getUint32(4, true);
		if (magic !== stub.magic || version !== stub.version) {
			throw new Error(`field bin header mismatch (got ${magic.toString(16)}/${version})`);
		}
		const n_freq = dv.getUint32(8, true);
		const n_port = dv.getUint32(12, true);
		const stride = dv.getUint32(16, true);
		const mask_off = 20;
		const mask = new Uint8Array(buf, mask_off, n_freq * n_port);
		// The mask block is zero-padded so the float block starts on a
		// 4-byte boundary — required by Float32Array's offset constraint.
		const mask_padded = (mask.byteLength + 3) & ~3;
		const floats_off = mask_off + mask_padded;
		const all_floats = new Float32Array(buf, floats_off);

		const out: (number[] | null)[][] = [];
		let cursor = 0;
		for (let fi = 0; fi < n_freq; fi++) {
			const row: (number[] | null)[] = [];
			for (let pi = 0; pi < n_port; pi++) {
				if (mask[fi * n_port + pi] === 0) {
					row.push(null);
				} else {
					row.push(Array.from(all_floats.subarray(cursor, cursor + stride)));
					cursor += stride;
				}
			}
			out.push(row);
		}
		return out;
	}

	/** Find the baked cell matching the source the editor just sent. */
	private find_cell(example: BakedExample, code: string): BakedCell | null {
		const eq = (a: string, b: string) => a === b || a.trimEnd() === b.trimEnd();
		const exact = example.cells.find((c) => eq(c.code, code));
		if (exact) return exact;
		// Fallback: trimmed match in case Notebook.serialize() added/dropped
		// a trailing newline.
		return example.cells.find((c) => c.code.trim() === code.trim()) ?? null;
	}

	execute(opts: ExecuteOptions): Promise<ExecuteResult> {
		return (async () => {
			opts.onStarted?.();
			let example: BakedExample;
			try {
				example = await this.load_example(opts.file);
			} catch (err) {
				opts.onStream?.('stderr', String(err));
				return { ok: false, error: { type: 'StaticDemoError', message: String(err), traceback: '' } };
			}
			const cell = this.find_cell(example, opts.code);
			if (!cell) {
				opts.onStream?.('stderr', '[static-demo] no baked record for this cell');
				return { ok: false, error: { type: 'StaticDemoError', message: 'no cell match', traceback: '' } };
			}

			for (const s of cell.stream_lines) opts.onStream?.(s.stream, s.line);

			for (const ev of cell.display_events) {
				if (ev.kind === 'error') {
					if (ev.error) opts.onStream?.('stderr', `${ev.error.type}: ${ev.error.message}`);
					continue;
				}
				let payload = ev.payload ?? {};
				if (ev.kind === 'result') {
					const f = (payload as Record<string, unknown>).fields as BakedFieldsStub | undefined | null;
					if (f && (f as BakedFieldsStub).$bin) {
						try {
							const hydrated = await this.hydrate_fields(f as BakedFieldsStub);
							payload = { ...payload, fields: hydrated };
						} catch (err) {
							opts.onStream?.('stderr', `[static-demo] field load failed: ${err}`);
						}
					}
				}
				opts.onDisplay?.(ev.kind, payload, ev.name);
			}

			if (cell.status === 'error') {
				return { ok: false, error: cell.error };
			}
			return { ok: true };
		})();
	}

	reset(_file: string) {
		/* no-op in static mode — the namespace is whatever was baked */
	}
}

let _singleton: KernelClient | StaticKernelClient | null = null;
export function get_kernel(): KernelClient | StaticKernelClient {
	if (!_singleton) {
		_singleton = IS_STATIC_MODE ? new StaticKernelClient() : new KernelClient();
	}
	return _singleton;
}
