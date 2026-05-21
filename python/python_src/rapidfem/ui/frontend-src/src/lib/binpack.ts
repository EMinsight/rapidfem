/**
 * Binary payload unpacking — the frontend counterpart of
 * `rapidfem.ui.binpack` (Python).
 *
 * Display-event payloads arrive with their bulk numeric arrays lifted into
 * two binary buffers — `geo` (mesh / geometry) and `field` (field and
 * trajectory data) — each array replaced by a `$bin` reference. This module
 * resolves those references back into typed arrays.
 *
 * The two buffers have different lifetimes: `geo` is resolved as soon as a
 * payload is delivered (the 3-D view needs it); `field` is resolved lazily,
 * only once the field viewer asks for it — see `resolveFieldRefs`.
 */

const BIN_MAGIC = 0x52464250; // "RFBP"
const BIN_VERSION = 2;

/** A `$bin` reference left in a payload in place of a bulk array. */
export interface BinRef {
	$bin: 'geo' | 'field';
	dtype: 'f32' | 'i32' | 'u16' | 'u8';
	off: number;
	n: number;
	/** Set on a frequency-domain field block. */
	kind?: 'fields' | 'frames';
	n_freq?: number;
	n_port?: number;
	stride?: number;
	mask?: number[];
	n_snap?: number;
	n_points?: number;
}

export function isBinRef(x: unknown): x is BinRef {
	return (
		typeof x === 'object' &&
		x !== null &&
		'$bin' in x &&
		((x as BinRef).$bin === 'geo' || (x as BinRef).$bin === 'field')
	);
}

/** Verify the 8-byte buffer header; throw on a stale or foreign blob. */
export function checkBinHeader(buf: ArrayBuffer, label: string): void {
	if (buf.byteLength < 8) throw new Error(`${label}: buffer too short`);
	const dv = new DataView(buf);
	const magic = dv.getUint32(0, true);
	const version = dv.getUint32(4, true);
	if (magic !== BIN_MAGIC || version !== BIN_VERSION) {
		throw new Error(
			`${label}: binary header mismatch ` +
				`(magic ${magic.toString(16)}, version ${version})`
		);
	}
}

/** A `dtype`-correct typed-array view onto `buf` at `ref.off`. */
function typedView(buf: ArrayBuffer, ref: BinRef): Float32Array | Int32Array | Uint16Array | Uint8Array {
	switch (ref.dtype) {
		case 'f32':
			return new Float32Array(buf, ref.off, ref.n);
		case 'i32':
			return new Int32Array(buf, ref.off, ref.n);
		case 'u16':
			return new Uint16Array(buf, ref.off, ref.n);
		case 'u8':
			return new Uint8Array(buf, ref.off, ref.n);
		default:
			throw new Error(`binpack: unknown dtype ${(ref as BinRef).dtype}`);
	}
}

/** Resolve a plain-array `$bin` ref to a typed-array view. */
export function resolveArray(buf: ArrayBuffer, ref: BinRef): Float32Array | Int32Array | Uint16Array | Uint8Array {
	return typedView(buf, ref);
}

/** Resolve a `kind:"fields"` ref to the `[n_freq][n_port]` nested shape the
 *  field viewer consumes — `null` where the presence mask is 0. */
export function resolveFields(buf: ArrayBuffer, ref: BinRef): (number[] | null)[][] {
	const n_freq = ref.n_freq ?? 0;
	const n_port = ref.n_port ?? 0;
	const stride = ref.stride ?? 0;
	const mask = ref.mask ?? [];
	const all = new Float32Array(buf, ref.off, ref.n);
	const out: (number[] | null)[][] = [];
	let cursor = 0;
	for (let fi = 0; fi < n_freq; fi++) {
		const row: (number[] | null)[] = [];
		for (let pi = 0; pi < n_port; pi++) {
			if (mask[fi * n_port + pi] === 0) {
				row.push(null);
			} else {
				row.push(Array.from(all.subarray(cursor, cursor + stride)));
				cursor += stride;
			}
		}
		out.push(row);
	}
	return out;
}

/** Resolve a `kind:"frames"` ref to the `[n_snap][n_points]` nested shape.
 *  For a time-domain trajectory the per-frame row length (`n_points`) is the
 *  unique-node count of the trajectory's DG-corner mesh. */
export function resolveFrames(buf: ArrayBuffer, ref: BinRef): number[][] {
	const n_snap = ref.n_snap ?? 0;
	const n_points = ref.n_points ?? 0;
	const all = new Uint16Array(buf, ref.off, ref.n);
	const out: number[][] = [];
	for (let s = 0; s < n_snap; s++) {
		out.push(Array.from(all.subarray(s * n_points, (s + 1) * n_points)));
	}
	return out;
}

/** Resolve any `$bin` ref against the buffer it names. A plain ref yields a
 *  typed array; a `fields` / `frames` ref yields the nested shape. */
export function resolveRef(
	ref: BinRef,
	geo: ArrayBuffer | null,
	field: ArrayBuffer | null,
): unknown {
	const buf = ref.$bin === 'geo' ? geo : field;
	if (!buf) throw new Error(`binpack: ${ref.$bin} buffer not loaded`);
	if (ref.kind === 'fields') return resolveFields(buf, ref);
	if (ref.kind === 'frames') return resolveFrames(buf, ref);
	return resolveArray(buf, ref);
}

/** Walk a payload and resolve every `geo`-buffer ref in place. `field` refs
 *  are left untouched — the field viewer resolves those lazily. */
export function resolveGeoRefs(payload: Record<string, unknown>, geo: ArrayBuffer): void {
	walk(payload, (ref) => (ref.$bin === 'geo' ? resolveRef(ref, geo, null) : ref));
}

/** Walk a payload and resolve every `field`-buffer ref in place. */
export function resolveFieldRefs(payload: Record<string, unknown>, field: ArrayBuffer): void {
	walk(payload, (ref) => (ref.$bin === 'field' ? resolveRef(ref, null, field) : ref));
}

/** Recursively replace `$bin` refs reached from `node` via `fn`. */
function walk(node: unknown, fn: (ref: BinRef) => unknown): void {
	if (Array.isArray(node)) {
		for (let i = 0; i < node.length; i++) {
			const v = node[i];
			if (isBinRef(v)) node[i] = fn(v);
			else if (typeof v === 'object' && v !== null) walk(v, fn);
		}
	} else if (typeof node === 'object' && node !== null) {
		for (const k of Object.keys(node as Record<string, unknown>)) {
			const v = (node as Record<string, unknown>)[k];
			if (isBinRef(v)) (node as Record<string, unknown>)[k] = fn(v);
			else if (typeof v === 'object' && v !== null) walk(v, fn);
		}
	}
}
