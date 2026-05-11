/**
 * `MeshSpec` — TypeScript mirror of the Rust `rapidfem_mesher::MeshSpec`.
 * Build one of these in JS, hand it to the WASM mesher via the worker.
 *
 * Coordinates are in METERS.
 */

export interface DielectricSlab {
	name: string;
	z_bottom: number;
	z_top: number;
}

export interface ConductorPolygon {
	name: string;
	xy: [number, number][];      // CCW outline in xy
	z_bottom: number;
	z_top: number;
}

export interface VerticalPort {
	name: string;
	xy_a: [number, number];
	xy_b: [number, number];
	z_bottom: number;
	z_top: number;
}

/** PML wrap around the inner simulation domain. Mesher extends footprint
 *  by `thickness` in xy and adds `thickness`-height slabs above/below the
 *  dielectric stack; FEM applies stretched-coordinate Maxwell to those
 *  cells. Replaces the need for big inner padding to push 1st-order ABC
 *  reflections out of the near-field region. */
export interface PmlSpec {
	thickness: number;
	er_base?: number;     // default 1
	ur_base?: number;     // default 1
	exponent?: number;    // default 1.5
	delta_max?: number;   // default 8
}

export interface MeshSpec {
	footprint_min: [number, number];
	footprint_max: [number, number];
	dielectrics: DielectricSlab[];
	conductors: ConductorPolygon[];
	ports: VerticalPort[];
	abc_tag: string;
	maxh: number;
	/** Optional separate z-step (vertical refinement). Defaults to `maxh`. */
	z_maxh?: number;
	pml?: PmlSpec;
}
