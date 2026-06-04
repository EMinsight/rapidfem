/**
 * HTTP-based kernel client.
 *
 * Each notebook file gets a long-lived worker subprocess (managed by
 * rapidfem.ui.runner). Cell execution is:
 *   1. POST /api/cell/run  → server kicks off subprocess, returns cell_id
 *   2. POST /api/cell/poll → long-polls (100 ms) for stream/display/error/done
 *      events until `done: true` lands in the response.
 *
 * The previous WS-based path is gone — see commits leading to
 * feature/subprocess-worker for the rationale (Werkzeug+wsproto+deflate
 * frame-encoding bugs that bit on >64 KiB messages).
 */

import {
	api_base, type GeometryPayload, type MeshPayload, type PythonError,
	type TdResultPayload, type TdTimeSeriesPayload, type TdTrajectoryPayload,
} from './api';

export type StreamKind = 'stdout' | 'stderr';

/** Display-event `kind`s the kernel forwards to `onDisplay`. */
export type DisplayKind =
	| 'geometry' | 'mesh' | 'result' | 'sweep_point'
	| 'td_result' | 'td_timeseries' | 'td_transfer' | 'td_trajectory';

export type KernelEvent =
	| { type: 'stream'; stream: StreamKind; value: string }
	| { type: 'display'; kind: 'geometry'; name: string; payload: GeometryPayload }
	| { type: 'display'; kind: 'mesh'; name: string; payload: MeshPayload }
	| { type: 'display'; kind: 'result'; name: string; payload: SolveResultPayload }
	| { type: 'display'; kind: 'sweep_point'; name: string; payload: SweepPointPayload }
	| { type: 'display'; kind: 'td_result'; name: string; payload: TdResultPayload }
	| { type: 'display'; kind: 'td_timeseries'; name: string; payload: TdTimeSeriesPayload }
	| { type: 'display'; kind: 'td_transfer'; name: string; payload: TdTimeSeriesPayload }
	| { type: 'display'; kind: 'td_trajectory'; name: string; payload: TdTrajectoryPayload }
	| { type: 'display'; kind: 'error'; name: string; error: PythonError }
	| { type: 'error'; id?: string; error: string; traceback?: string }
	| { type: 'done'; id: string; ok: boolean }
	| { type: 'reset-ack' }
	| { type: 'worker-exit' };

export interface SolveResultPayload {
	frequencies: number[];
	sparams: number[][][][];
	n_driven: number;
	n_freq: number;
	n_dofs: number;
	n_tets: number;
	/** True when this is an eigenmode result — `frequencies[i]` is the i-th
	 *  resonant frequency, `sparams` is empty, the field slider becomes a
	 *  mode-index slider. */
	eigenmode?: boolean;
	/** Per-mode Q factor (only present when eigenmode=true). */
	q_factors?: number[];
	solve_time_s: number;
	fields?: (number[] | null)[][];
	/** Conduction current density J = σE per (freq, port), same shape as `fields`. */
	fields_j?: (number[] | null)[][] | null;
	/** Magnetic field H = ∇×E / (jωμ) per (freq, port), same shape as `fields`. */
	fields_h?: (number[] | null)[][] | null;
	/** Driven-sweep fields are fetched on demand (binary) via /api/field rather
	 *  than inlined; this carries the available shape instead of the arrays. */
	field_meta?: { n_freq: number; n_port: number; channels: string[]; lazy?: boolean };
}

/** One frequency's S-matrix, streamed live during a sweep. `s[obs][exc]` is
 *  `[re, im]`. The viewer appends these incrementally (no full-result rebuild). */
export interface SweepPointPayload {
	freq_idx: number;
	freq: number;
	s: number[][][];
}

export interface ExecuteOptions {
	cell_id: string;
	file: string;
	code: string;
	reset?: boolean;
	onStarted?: () => void;
	onStream?: (stream: StreamKind, line: string) => void;
	onDisplay?: (kind: DisplayKind, payload: unknown, name: string) => void;
}

export interface ExecuteResult {
	ok: boolean;
	error?: PythonError;
}

async function post_json<T>(path: string, body: object): Promise<T> {
	const res = await fetch(api_base() + path, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
	});
	if (!res.ok) {
		const text = await res.text().catch(() => '');
		throw new Error(`${path}: HTTP ${res.status} ${text.slice(0, 200)}`);
	}
	return res.json() as Promise<T>;
}

export class KernelClient {
	async execute(opts: ExecuteOptions): Promise<ExecuteResult> {
		console.log('[kernel] execute', { cell_id: opts.cell_id, file: opts.file, code_len: opts.code.length });
		try {
			const run = await post_json<{ ok: boolean; cell_id?: string; error?: string }>(
				'/api/cell/run',
				{
					file: opts.file,
					code: opts.code,
					reset: !!opts.reset,
					cell_id: opts.cell_id,
				},
			);
			if (!run.ok || !run.cell_id) {
				console.warn('[kernel] /api/cell/run rejected', run);
				return {
					ok: false,
					error: { type: 'RunError', message: run.error ?? 'cell-run failed', traceback: '' },
				};
			}
			opts.onStarted?.();
			const r = await this.drain_until_done(opts);
			if (!r.ok) console.warn('[kernel] execute returning ok=false', r);
			return r;
		} catch (e) {
			console.error('[kernel] execute caught:', e);
			return {
				ok: false,
				error: { type: 'NetworkError', message: String(e), traceback: '' },
			};
		}
	}

	private async drain_until_done(opts: ExecuteOptions): Promise<ExecuteResult> {
		const target = opts.cell_id;
		let last_error: PythonError | undefined;
		// Poll loop. The server's /api/cell/poll long-polls 100 ms so this
		// stays cheap and responsive.
		while (true) {
			let resp: { messages: KernelEvent[]; done: boolean };
			try {
				resp = await post_json<{ messages: KernelEvent[]; done: boolean }>(
					'/api/cell/poll',
					{ file: opts.file },
				);
			} catch (e) {
				return {
					ok: false,
					error: { type: 'NetworkError', message: String(e), traceback: '' },
				};
			}
			for (const evt of resp.messages) {
				if (evt.type === 'stream') {
					// Server sends raw chunks; split into lines so the
					// notebook log panel renders cleanly.
					for (const line of evt.value.split('\n')) {
						if (line.length > 0) opts.onStream?.(evt.stream, line);
					}
				} else if (evt.type === 'display') {
					if (evt.kind === 'error') {
						opts.onStream?.('stderr', `${evt.error.type}: ${evt.error.message}`);
					} else {
						// onDisplay callbacks touch reactive state (mesh, S-params,
						// fields). A throw in there used to bubble all the way up
						// and mark the cell failed — but the cell itself already
						// finished cleanly on the worker. Log + skip instead so
						// the user sees the diagnostic, the cell stays OK, and
						// the next display events keep flowing.
						try {
							opts.onDisplay?.(evt.kind, evt.payload, evt.name);
						} catch (err) {
							console.error('[kernel] onDisplay threw:', err, evt);
							opts.onStream?.('stderr',
								`[ui] display "${evt.kind}" failed to render: ${err}`);
						}
					}
				} else if (evt.type === 'error') {
					last_error = {
						type: 'ExecError',
						message: evt.error,
						traceback: evt.traceback ?? '',
					};
					opts.onStream?.('stderr', evt.error);
					if (evt.traceback) opts.onStream?.('stderr', evt.traceback);
				} else if (evt.type === 'done') {
					if (evt.id === target) {
						console.log('[kernel] cell done', { cell_id: target, ok: evt.ok, has_error: !!last_error });
						return { ok: evt.ok, error: last_error };
					}
					console.warn('[kernel] done event for non-matching cell', {
						received_id: evt.id, target,
					});
					opts.onStream?.('stderr',
						`[ui] WARNING: received done for cell_id="${evt.id}" but expected "${target}" — cell may show stale status`);
				} else if (evt.type === 'worker-exit') {
					console.warn('[kernel] worker exited');
					opts.onStream?.('stderr', '[ui] worker process exited');
					return {
						ok: false,
						error: { type: 'WorkerExit', message: 'Worker process exited', traceback: '' },
					};
				}
			}
			// If the server says we're done but we never saw our own `done`,
			// something raced — bail with whatever error we caught.
			if (resp.done && resp.messages.every((e) => e.type !== 'done' || e.id !== target)) {
				return last_error
					? { ok: false, error: last_error }
					: { ok: true };
			}
			// Empty-but-not-done poll → idle wait until next tick. The server
			// already blocks ~100 ms so we don't add a JS-side delay here.
		}
	}

	async reset(file: string): Promise<void> {
		try {
			await post_json('/api/cell/reset', { file });
		} catch (e) {
			console.warn('[kernel] reset failed:', e);
		}
	}

	/** Send SIGINT to the worker subprocess — the cell-run's `exec` will
	 *  raise `KeyboardInterrupt`, propagate to the error path, and emit an
	 *  `error` event back through the normal poll stream. */
	async interrupt(file: string): Promise<boolean> {
		try {
			const r = await post_json<{ ok: boolean }>('/api/cell/interrupt', { file });
			return !!r.ok;
		} catch (e) {
			console.warn('[kernel] interrupt failed:', e);
			return false;
		}
	}

	/** No binary field sidecar in live mode — the WebSocket delivers field
	 *  arrays inline. Present so callers can treat both kernels uniformly. */
	async fieldBuffer(_file: string): Promise<ArrayBuffer | null> {
		return null;
	}
}

// ── Static-demo replay client ──────────────────────────────────────────
//
// Used when the frontend is built with VITE_STATIC_MODE=1. Loads the bake
// artefacts produced by `scripts/bake_demo.py` from `static/demo/` and
// replays them through the same execute() callbacks as the live WS path,
// so consumers (Cell, Notebook, +page.svelte) don't have to branch.

import { IS_STATIC_MODE, DEMO_BASE } from './static_mode';
import { checkBinHeader, resolveGeoRefs, resolveFieldRefs } from './binpack';

interface BakedDisplayEvent {
	kind: DisplayKind | 'error';
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
	/** Cached `<name>.geo.bin` / `<name>.field.bin` buffers, keyed
	 *  `"<name>.<geo|field>"`. `null` once it is known the example carries
	 *  no buffer of that kind. */
	private blobs = new Map<string, Promise<ArrayBuffer | null>>();

	/** No worker process in static demo mode - nothing to interrupt. */
	async interrupt(_file: string): Promise<boolean> {
		return false;
	}

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

	/** Fetch a `<name>.geo.bin` / `<name>.field.bin` sidecar, cached.
	 *  Resolves to `null` when the example carries no buffer of that kind
	 *  (the manifest does not list it). */
	private load_blob(name: string, suffix: 'geo' | 'field'): Promise<ArrayBuffer | null> {
		const key = `${name}.${suffix}`;
		let p = this.blobs.get(key);
		if (p) return p;
		p = (async () => {
			const fname = `${name}.${suffix}.bin`;
			const m = await this.load_manifest();
			const entry = m.examples.find((e) => e.name === name);
			if (entry && Array.isArray(entry.bin_files) && !entry.bin_files.includes(fname)) {
				return null;
			}
			const r = await fetch(`${DEMO_BASE}${fname}`);
			if (!r.ok) return null;
			const buf = await r.arrayBuffer();
			checkBinHeader(buf, fname);
			return buf;
		})();
		this.blobs.set(key, p);
		return p;
	}

	/** The `field` buffer of an example — fetched lazily, only when the
	 *  field viewer (or a trajectory) actually asks for it. `null` in
	 *  examples without field data, or in any non-static kernel. */
	async fieldBuffer(file: string): Promise<ArrayBuffer | null> {
		const ex = await this.load_example(file);
		return this.load_blob(ex.name, 'field');
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

			// The `geo` buffer (mesh / geometry) is resolved up front — the
			// 3-D view needs it immediately. `field`-buffer refs are left in
			// the payload; the field viewer hydrates those lazily through
			// `fieldBuffer()`, so an example browsed for geometry + S-params
			// never fetches its field data at all.
			const geo = await this.load_blob(example.name, 'geo');
			let field: ArrayBuffer | null = null;
			let field_loaded = false;
			for (const ev of cell.display_events) {
				if (ev.kind === 'error') {
					if (ev.error) opts.onStream?.('stderr', `${ev.error.type}: ${ev.error.message}`);
					continue;
				}
				const payload = ev.payload ?? {};
				if (geo) {
					try {
						resolveGeoRefs(payload, geo);
					} catch (err) {
						opts.onStream?.('stderr', `[static-demo] geometry load failed: ${err}`);
					}
				}
				// A trajectory's point cloud IS the displayed content, so its
				// field-buffer refs are resolved eagerly here, the same way
				// geo refs are. FD field channels stay lazy (fieldBuffer()).
				if (ev.kind === 'td_trajectory') {
					if (!field_loaded) {
						field = await this.load_blob(example.name, 'field');
						field_loaded = true;
					}
					if (field) {
						try {
							resolveFieldRefs(payload, field);
						} catch (err) {
							opts.onStream?.('stderr', `[static-demo] trajectory load failed: ${err}`);
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
