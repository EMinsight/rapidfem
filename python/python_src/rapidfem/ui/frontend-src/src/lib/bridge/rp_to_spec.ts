/**
 * Bridge: rapidpassives layout → rapidfem GeometrySpec.
 *
 * Takes the artifacts a rapidpassives generator produces (LayerMap of 2D
 * polygons-per-layer, a ProcessStack, port markers) and converts them into a
 * neutral GeometrySpec that rapidfem's mesher consumes.
 *
 * rapidpassives uses µm internally; we convert to meters here so the spec
 * is unit-consistent with the FEM solver.
 *
 * Types are kept structural (no `import type from $rapidpassives/...`) so
 * this bridge is portable to any caller that produces compatible shapes.
 */
import type { GeometrySpec, SpecLayer, SpecPolygon, SpecPort } from './spec';

export interface RpPolygon { x: number[]; y: number[] }   // µm
export interface RpStackLayer {
	id: string;
	name: string;
	type: 'metal' | 'via' | 'substrate';
	z: number;            // µm
	thickness: number;    // µm
	gdsLayers: string[];
}
export interface RpStack {
	name: string;
	layers: RpStackLayer[];
	substrateThickness: number;       // µm
	oxideEr: number;
	substrateRho: number;             // ohm·cm
	substrateEr: number;
}
export interface RpPort {
	name: string;
	x: number;            // µm
	y: number;            // µm
	/** Layer id of the trace metal (e.g. "m3") — looked up in the stack to
	 *  find the matching FEM layer name. */
	layer_id?: string;
	/** Layer id of the ground reference. */
	gnd_layer_id?: string;
}

export interface RpBridgeOptions {
	/** Air padding around the layout bbox [m]. Defaults to 50 µm xy / 30 µm z. */
	air_padding_xy?: number;
	air_padding_z?: number;
	/** Mesh size [m]. */
	maxh?: number;
	frequencies_hz: number[];
	port_size?: number;       // [m]
	z0?: number;
}

const um = 1e-6;

/** Convert ohm·cm → S/m for substrate conductivity. */
function rho_cm_to_sigma(rho_ohm_cm: number): number {
	if (!rho_ohm_cm || rho_ohm_cm <= 0) return 0;
	// 1 Ω·cm = 0.01 Ω·m → σ = 1/(rho·0.01) = 100/rho S/m
	return 100 / rho_ohm_cm;
}

export function rp_to_spec(
	name: string,
	layers: Record<string, RpPolygon[] | undefined>,
	stack: RpStack,
	ports: { name: string; x: number; y: number; layer_id?: string; gnd_layer_id?: string }[],
	opts: RpBridgeOptions
): GeometrySpec {
	// 1. Translate the stack — pick metal/via/substrate from RpStack and add
	//    the implicit oxide slabs that fill the gaps between metals.
	const sorted = [...stack.layers].sort((a, b) => a.z - b.z);
	const spec_layers: SpecLayer[] = [];

	// Substrate (rapidpassives substrate layer is z=0..thickness in µm; FEM
	// convention has substrate BELOW the metal stack at z<0, so we shift)
	const sub = sorted.find((l) => l.type === 'substrate');
	const sub_top = sub ? (sub.z + sub.thickness) * um : 0;
	if (sub) {
		spec_layers.push({
			name: 'substrate',
			type: 'substrate',
			z: sub_top - sub.thickness * um,
			thickness: sub.thickness * um,
			er: stack.substrateEr,
			conductivity: rho_cm_to_sigma(stack.substrateRho)
		});
	}

	// Metals + vias above substrate top
	const metals: SpecLayer[] = [];
	for (const l of sorted) {
		if (l.type === 'substrate') continue;
		metals.push({
			name: l.id,
			type: l.type === 'via' ? 'via' : 'metal',
			z: l.z * um,
			thickness: l.thickness * um
		});
	}
	// Oxide blob spanning from substrate top to top metal top
	if (metals.length > 0) {
		const top = Math.max(...metals.map((m) => m.z + m.thickness));
		spec_layers.push({
			name: 'oxide',
			type: 'oxide',
			z: sub_top,
			thickness: top - sub_top,
			er: stack.oxideEr
		});
	}
	spec_layers.push(...metals);

	// 2. Polygons by FEM layer name. rapidpassives' layer names (windings,
	//    crossings, ...) are mapped to stack layer ids via `gdsLayers`.
	const gdsToStackId = new Map<string, string>();
	for (const sl of stack.layers) {
		for (const gl of sl.gdsLayers) gdsToStackId.set(gl, sl.id);
	}
	const spec_polys: SpecPolygon[] = [];
	let xmin = +Infinity, ymin = +Infinity, xmax = -Infinity, ymax = -Infinity;
	for (const [layer_name, polys] of Object.entries(layers)) {
		if (!polys) continue;
		const stack_id = gdsToStackId.get(layer_name);
		if (!stack_id) continue;     // layer not in stack → ignore
		for (const p of polys) {
			const xy: number[] = [];
			for (let i = 0; i < p.x.length; i++) {
				const x = p.x[i] * um;
				const y = p.y[i] * um;
				xy.push(x, y);
				if (x < xmin) xmin = x; if (x > xmax) xmax = x;
				if (y < ymin) ymin = y; if (y > ymax) ymax = y;
			}
			spec_polys.push({ layer: stack_id, xy });
		}
	}

	// 3. Ports. If layer_id / gnd_layer_id missing, default to top metal /
	//    bottom metal in the stack.
	const top_metal = metals.length > 0 ? metals[metals.length - 1].name : '';
	const bottom_metal = metals.length > 0 ? metals[0].name : '';
	const spec_ports: SpecPort[] = ports.map((p) => ({
		name: p.name,
		x: p.x * um,
		y: p.y * um,
		layer: p.layer_id ?? top_metal,
		gnd_layer: p.gnd_layer_id ?? bottom_metal,
		size: opts.port_size ?? 6e-6,
		direction: [0, 0, 1],
		z0: opts.z0 ?? 50
	}));

	const air_xy = opts.air_padding_xy ?? 50e-6;
	const air_z = opts.air_padding_z ?? 30e-6;

	return {
		name,
		stack: spec_layers,
		polygons: spec_polys,
		ports: spec_ports,
		boundary: { air_padding_xy: air_xy, air_padding_z: air_z, abc: 'B' },
		frequencies_hz: opts.frequencies_hz,
		maxh: opts.maxh ?? 15e-6
	};
}
