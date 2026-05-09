/** Demo example registry. msh + toml are pre-built by wasm-demo/scripts/. */

export interface DemoExample {
	id: string;
	label: string;
	description: string;
	msh_url: string;     // path under /examples/
	toml_url: string;
	frequencies_hz: number[];
	extract_l: boolean;
}

export const EXAMPLES: Record<string, DemoExample> = {
	wr90: {
		id: 'wr90',
		label: 'WR-90 waveguide',
		description: 'Rectangular hollow waveguide, dominant TE10 mode, fundamental EM benchmark.',
		msh_url: '/examples/wr90_straight.msh',
		toml_url: '/examples/wr90.toml',
		frequencies_hz: [9e9, 9.25e9, 9.5e9, 9.75e9, 10e9, 10.25e9, 10.5e9, 10.75e9, 11e9],
		extract_l: false
	},
	microstrip: {
		id: 'microstrip',
		label: 'Sky130 microstrip',
		description: '200 µm × 5 µm trace on met5 over a continuous li1 ground strip — true microstrip topology.',
		msh_url: '/examples/microstrip.msh',
		toml_url: '/examples/microstrip.toml',
		frequencies_hz: [1e9, 2e9, 3e9, 4e9, 5e9],
		extract_l: true
	},
	spiral: {
		id: 'spiral',
		label: 'Sky130 spiral inductor',
		description: 'Octagonal 1-turn 80 µm spiral on met5 with local li1 ground patches under each port. Sweep captures L_eq(f) self-resonance.',
		msh_url: '/examples/spiral.msh',
		toml_url: '/examples/spiral.toml',
		frequencies_hz: [1e9, 10e9, 30e9, 50e9, 80e9, 110e9, 140e9, 170e9, 200e9, 230e9, 250e9],
		extract_l: true
	}
};
