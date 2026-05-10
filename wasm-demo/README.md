# rapidfem WASM demo

Sky130 microstrip solved entirely client-side: TypeScript builds a `MeshSpec`,
the in-browser mesher (`rapidfem-mesher`) generates a tagged tet mesh, the FEM
kernel (`rapidfem`, compiled to WASM) runs the frequency sweep, and the
SvelteKit frontend plots S-parameters and renders the volumetric `|E(t)|`
field with a GPU-driven wave animation. No backend, no Python, no static
mesh files.

## Layout

- `src/lib.rs` — wasm-bindgen wrapper exposing `mesh_from_spec` (for the
  viewer) and `solve_from_spec` (full mesh + solve pipeline)
- `app/` — SvelteKit application
  - `src/lib/solver.worker.ts` — owns the WASM module; runs meshing, solving,
    and field point-cloud sampling off the main thread
  - `src/lib/wasm.ts` — main-thread RPC to the worker
  - `src/lib/components/MeshViewer.svelte` — WebGL2 3D viewer (Geometry /
    Wireframe / Field toggles, phase-animated point cloud)
  - `src/lib/examples.ts` — the demo spec (microstrip dimensions, PML
    configuration, frequencies, metrics)

## Build

```bash
# from the repo root
wasm-pack build wasm-demo --target web --release --out-dir pkg
cp wasm-demo/pkg/* wasm-demo/app/static/pkg/
```

The WASM bundle (~800 KB compressed) lands in `wasm-demo/app/static/pkg/`
where Vite serves it as a static asset.

## Run

```bash
cd wasm-demo/app
npm install        # first time
npm run dev
```

Open `http://localhost:5173/`. Pick **Sky130 microstrip** from the example
dropdown, hit **Run sweep**, then explore the 3D View — toggle Geometry,
Wireframe, or Field. With Field on, the **Wave** checkbox starts a
time-domain animation; the speed slider controls cycles-per-second.

## Pipeline

```
build_spec() in TS
   │   MeshSpec (JSON)
   ▼
mesh_from_spec()         ← WASM (rapidfem-mesher)
   │   tagged tet mesh
   ▼
solve_from_spec()        ← WASM (rapidfem)
   │   S-params + per-node phasor terms (A, B, C)
   ▼
3D viewer + plots
```

The shader composes `|E(t)|² = A·cos²(ωt) + B·sin²(ωt) − 2C·cos·sin` per
vertex every frame from the static (A, B, C) attribute and a single phase
uniform — no per-frame CPU work, no field re-upload during animation.

## Constraints

WASM builds use the pure-Rust `faer` sparse LU instead of MKL PARDISO. The
32-bit linear-memory address space caps usable mesh size around ~50–60 k
DOFs (LU fill-in is the bottleneck, not the matrix itself). The
`live_microstrip` example is tuned to fit comfortably under that ceiling
with PML active.

PML uses a single tet layer across its thickness by default to stay within
the DOF budget — boost `pml.n_layers` in `MeshSpec` for cleaner inner-face
matching when running natively or once Memory64 is widely available.
