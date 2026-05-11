/** Demo example registry. Every example builds its `MeshSpec` directly in
 *  TypeScript; the rapidfem-mesher WASM crate then meshes + solves end-to-end
 *  in the browser. No pre-baked .msh, no Python bridge. */

export type Metric =
	| 'L_eq'    // π-equivalent series inductance — meaningful for inductors
	| 'Q'       // quality factor at port 1 — meaningful for inductors / resonators
	| 'Z0';     // characteristic impedance — meaningful for transmission lines

export interface DemoExample {
	id: string;
	label: string;
	description: string;
	build_spec: () => {
		spec: object;
		materials?: Record<string, { er: number; conductivity?: number; tand?: number }>;
	};
	frequencies_hz: number[];
	metrics: Metric[];
}

export const EXAMPLES: Record<string, DemoExample> = {
	microstrip: {
		id: 'microstrip',
		label: 'Sky130 microstrip',
		description:
			'200×5 µm met5 trace over a continuous li1 ground strip. Spec is built in TS, meshed in WASM via rapidfem-mesher (spade CDT + 2.5D extrusion + refinement), and solved with the in-browser FEM kernel.',
		frequencies_hz: [1e9, 2e9, 3e9, 4e9, 5e9],
		metrics: ['Z0'],
		build_spec: () => {
			const um = 1e-6;
			const trace_l = 200 * um;
			const trace_w = 5 * um;
			// PML wraps the inner domain on every side, so the inner padding
			// only needs to clear the near-field where E falls off rapidly
			// (~few × trace_w). The PML then absorbs whatever radiates out.
			// Mesh sizing: total DOFs ≈ tet_count × 7.
			// WASM 32-bit heap caps us around ~60k DOFs (faer sparse LU peak).
			// Knobs:
			//   pad      — clearance between trace and PML inner face (near-field gap).
			//              Smaller → tighter mesh but PML sees stronger near-field.
			//   pml_t    — PML thickness; absorption strength scales with thickness × δmax.
			//   maxh     — global tet edge target; refinement still concentrates at trace.
			//   n_layers — PML cells across thickness; 2 is the cheap-but-graded default.
			const pad = 30 * um;
			const air_h = 30 * um;
			const sub_h = 15 * um;
			const pml_t = 15 * um;
			return {
				spec: {
					footprint_min: [-trace_l / 2 - pad, -trace_w / 2 - pad],
					footprint_max: [trace_l / 2 + pad, trace_w / 2 + pad],
					dielectrics: [
						{ name: 'substrate', z_bottom: -sub_h, z_top: 0 },
						{ name: 'oxide', z_bottom: 0, z_top: 5.625 * um },
						{ name: 'air', z_bottom: 5.625 * um, z_top: 5.625 * um + air_h }
					],
					conductors: [
						{
							name: 'met5',
							xy: [
								[-trace_l / 2, -trace_w / 2], [trace_l / 2, -trace_w / 2],
								[trace_l / 2, trace_w / 2], [-trace_l / 2, trace_w / 2]
							],
							z_bottom: 4.365 * um,
							z_top: 5.625 * um
						},
						{
							name: 'li1_gnd',
							xy: [
								[-trace_l * 0.6, -trace_w * 5], [trace_l * 0.6, -trace_w * 5],
								[trace_l * 0.6, trace_w * 5], [-trace_l * 0.6, trace_w * 5]
							],
							z_bottom: 0,
							z_top: 0.1 * um
						}
					],
					ports: [
						{
							name: 'p1',
							xy_a: [-trace_l / 2, -trace_w / 2],
							xy_b: [-trace_l / 2, trace_w / 2],
							z_bottom: 0.1 * um,
							z_top: 4.365 * um
						},
						{
							name: 'p2',
							xy_a: [trace_l / 2, -trace_w / 2],
							xy_b: [trace_l / 2, trace_w / 2],
							z_bottom: 0.1 * um,
							z_top: 4.365 * um
						}
					],
					abc_tag: 'abc',
					maxh: 35 * um,
					// Finer vertical refinement (decoupled from in-plane `maxh`
					// which is dominated by the 5µm trace). 15µm keeps the
					// total DOF count comfortably under the WASM heap budget
					// — substrate/air get 1-2 z-layers, thin metals stay 1.
					z_maxh: 15 * um,
					pml: { thickness: pml_t }
				},
				materials: {
					substrate: { er: 11.9, conductivity: 10 },
					oxide: { er: 4.2 },
					air: { er: 1.0 }
				}
			};
		}
	}
};
