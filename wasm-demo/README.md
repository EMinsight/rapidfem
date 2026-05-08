# rapidfem WASM demo

A small in-browser FEM demo: solve a WR-90 waveguide sweep entirely in WebAssembly.

## Build

```bash
cd wasm-demo
wasm-pack build --target web --release
```

Outputs to `pkg/`. The .wasm is ~470 KB.

## Run locally

The demo needs a static HTTP server (file:// won't load .wasm modules).

From the `wasm-demo/` directory, in two terminals:

```bash
# serve repo root so JS can find both pkg/ and web/
cd ..
python -m http.server 8000
```

Then open `http://localhost:8000/wasm-demo/web/index.html`.

Click **Run sweep** — fetches the bundled mesh + TOML, runs the FEM solve in WASM,
plots |S11| and |S21| across 9–11 GHz on a canvas.

## How it works

- `src/lib.rs` — wasm-bindgen wrapper, exposes a single `run_sweep(mesh_bytes, config_toml)`.
- `web/index.html` — minimal browser UI: fetch assets, call the WASM, plot results.
- The same `Simulation` API used by the Python bindings and the native CLI is called here.

## Limitations

WASM build excludes:
- **PARDISO** (uses faer LU instead, ~10× slower)
- **rayon** (single-threaded assembly)
- **VTK export** (not needed in browser)

For small problems (a few thousand DOFs) the speed is fine. For production sweeps,
use the native binary or Python bindings.
