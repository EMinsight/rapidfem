/**
 * GeometrySpec — JSON-serializable schema describing everything needed to
 * mesh and simulate an RFIC structure. The neutral handoff format between
 * rapidpassives' layout generator (LayerMap + ProcessStack + ports) and
 * rapidfem's mesh+solver pipeline.
 *
 * Coordinates are in METERS (rapidpassives uses µm internally — convert
 * before populating the spec). Layer z + thickness are also in METERS.
 *
 * The mesher (server-side rapidfem.bridge or future WASM-tetgen) consumes
 * this and emits a {.msh, .toml} pair the WASM solver eats.
 */

export interface SpecLayer {
	/** Human-readable + unique identifier (e.g. "met5", "li1", "substrate"). */
	name: string;
	type: 'metal' | 'via' | 'substrate' | 'oxide' | 'air';
	/** Bottom z [m]. */
	z: number;
	thickness: number;
	er?: number;
	tan_d?: number;
	conductivity?: number;
}

export interface SpecPolygon {
	/** Layer name (must match one in `layers`). */
	layer: string;
	/** Closed polygon vertex list, [x0,y0, x1,y1, ...] in METERS. CCW outer. */
	xy: number[];
}

export interface SpecPort {
	/** Unique port name; becomes the FEM physical group + S-param row label. */
	name: string;
	/** xy center [m]. */
	x: number;
	y: number;
	/** Trace metal layer the port-plate top edge lands on (e.g. "met5"). */
	layer: string;
	/** Reference ground layer (e.g. "li1"). */
	gnd_layer: string;
	/** Square footprint side length [m] of the extension pad and gnd patch. */
	size?: number;          // default 6e-6
	/** Lumped-port voltage direction in global coords (default Z+). */
	direction?: [number, number, number];
	/** Reference impedance [Ω], default 50. */
	z0?: number;
}

export interface SpecBoundary {
	/** Air padding around the layout footprint, in METERS. */
	air_padding_xy: number;
	air_padding_z: number;
	/** ABC type ("B" = first-order Sommerfeld). */
	abc: 'B';
}

export interface GeometrySpec {
	name: string;
	stack: SpecLayer[];
	polygons: SpecPolygon[];
	ports: SpecPort[];
	boundary: SpecBoundary;
	/** Frequencies [Hz]. */
	frequencies_hz: number[];
	/** Mesh size hint [m]. */
	maxh: number;
}

/** Convenience guard / validator. Throws on missing references. */
export function validate_spec(spec: GeometrySpec): void {
	const layer_names = new Set(spec.stack.map((l) => l.name));
	for (const p of spec.polygons) {
		if (!layer_names.has(p.layer)) throw new Error(`polygon refers to unknown layer "${p.layer}"`);
		if (p.xy.length < 6 || p.xy.length % 2 !== 0) throw new Error(`polygon "${p.layer}" has bad xy length ${p.xy.length}`);
	}
	for (const port of spec.ports) {
		if (!layer_names.has(port.layer)) throw new Error(`port "${port.name}" refers to unknown layer "${port.layer}"`);
		if (!layer_names.has(port.gnd_layer)) throw new Error(`port "${port.name}" refers to unknown gnd_layer "${port.gnd_layer}"`);
	}
	if (spec.frequencies_hz.length === 0) throw new Error('spec.frequencies_hz is empty');
}
