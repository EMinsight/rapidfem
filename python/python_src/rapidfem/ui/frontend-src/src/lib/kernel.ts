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

let _singleton: KernelClient | null = null;
export function get_kernel(): KernelClient {
	if (!_singleton) _singleton = new KernelClient();
	return _singleton;
}
