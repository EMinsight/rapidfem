// Central configuration for the RapidFEM documentation site.
// Single-package — all site content is defined here.

export const site = {
	name: 'RapidFEM',
	tagline: 'Frequency-domain electromagnetic FEM solver',
	description:
		'A frequency-domain electromagnetic FEM solver written in Rust, distributed as a Python package. Second-kind Nedelec edge elements, complex-symmetric sparse linear algebra, waveguide and lumped ports.'
};

// External links. `demo` points at the existing static notebook demo that
// already ships the pre-baked examples — the docs do not re-render them.
export const external = {
	github: 'https://github.com/milanofthe/rapidfem',
	pypi: 'https://pypi.org/project/rapidfem',
	demo: 'https://fem.rapidpassives.org'
};

export interface Feature {
	title: string;
	description: string;
}

export const features: Feature[] = [
	{
		title: 'Nedelec-2 Elements',
		description: '20 DOFs per tetrahedron — vector edge basis for the curl–curl form of Maxwell.'
	},
	{
		title: 'Excitations',
		description: 'Rectangular waveguide ports, lumped TEM ports, and absorbing boundaries of order 1 and 2.'
	},
	{
		title: 'Anisotropic PML',
		description: 'Stretched-coordinate perfectly matched layer for open-domain radiation problems.'
	},
	{
		title: 'Sparse Solvers',
		description: 'Pure-Rust faer LU baseline, optional MKL PARDISO, and Apple Accelerate on macOS.'
	},
	{
		title: 'Frequency Sweep',
		description: 'Assembles E/B once, refactors only the frequency-dependent K, reuses the symbolic pattern.'
	},
	{
		title: 'Eigenmode Solver',
		description: 'Shift-invert Lanczos on the complex-symmetric system for resonant mode analysis.'
	},
	{
		title: 'Adaptive Refinement',
		description: 'Residual error estimator with Dörfler marking, exports a size field for gmsh re-meshing.'
	},
	{
		title: 'Output Formats',
		description: 'Touchstone (.sNp), VTK field export, and far-field NFFT radiation patterns.'
	}
];

export interface InstallOption {
	name: string;
	command: string;
	note?: string;
}

export const installation: InstallOption[] = [
	{ name: 'pip — solver', command: 'pip install rapidfem' },
	{ name: 'pip — solver + local UI', command: 'pip install rapidfem[ui]' }
];

export interface QuickStart {
	title: string;
	description: string;
	code: string;
}

export const quickstart: QuickStart = {
	title: 'Python API',
	description:
		'Build geometry, attach materials and physics to entities, then run any number of analyses on the same Problem.',
	code: `import numpy as np
import rapidfem as rf

# Build geometry; attach materials + physics directly to entities
g = rf.Geometry(maxh=rf.lambda_maxh(f_max=12e9))
air = g.box(22.86e-3, 10.16e-3, 30e-3, position=(-11.43e-3, -5.08e-3, 0),
            material=rf.Air())

rf.RectWaveguidePort(air.faces.min(axis="z"))
rf.RectWaveguidePort(air.faces.max(axis="z"))
rf.PEC(*air.faces.unassigned)

g.mesh()

# Define the problem once, run any number of analyses on it
prob = rf.Problem(g)
result = prob.sweep(np.linspace(8e9, 12e9, 21))
print(result.frequencies.shape, result.sparams.shape)`
};

export interface ApiModule {
	name: string;
	description: string;
}

// Python modules whose docstrings the build pipeline extracts.
export const apiModules: ApiModule[] = [
	{ name: 'rapidfem', description: 'Top-level package — Geometry, Problem, and re-exported helpers' },
	{ name: 'rapidfem.geometry', description: 'Geometry construction, entities, face selection, meshing' },
	{ name: 'rapidfem.problem', description: 'Problem definition, frequency sweep, eigenmode, far-field' },
	{ name: 'rapidfem.materials', description: 'Material models — Air, Dielectric, lossy permittivity' },
	{ name: 'rapidfem.physics', description: 'Ports and boundary conditions — waveguide, lumped, PEC, ABC' },
	{ name: 'rapidfem.io', description: 'Result containers, Touchstone, VTK, and far-field export' },
	{ name: 'rapidfem.rfic', description: 'RFIC layout import — geometry from JSON layer stacks' }
];

// Sidebar navigation. API/Examples paths use the version when known.
export interface SidebarItem {
	title: string;
	path: string;
	icon?: string;
	external?: boolean;
}

export function getSidebarItems(version?: string): SidebarItem[] {
	const v = version || 'latest';
	return [
		{ title: 'Overview', path: '', icon: 'home' },
		{ title: 'API Reference', path: `${v}/api`, icon: 'braces' }
	];
}
