# Static Web Demo

A read-only build of the notebook UI with pre-baked example outputs,
deployed to `fem.rapidpassives.org` via GitHub Pages.

## How it works

1. `scripts/bake_demo.py` walks `python/python_src/rapidfem/examples/*.py`,
   splits each file at `# %%` markers, runs every cell through the live
   kernel pipeline (gmsh, FEM solve, field sampling), and dumps the
   captured display events + stdout/stderr to
   `python/python_src/rapidfem/ui/frontend-src/static/demo/<name>.json`.
   Per-node field arrays are extracted to Float32 `.bin` sidecars to
   keep JSON small.

2. The SvelteKit frontend is built with `VITE_STATIC_MODE=1`. At runtime
   `lib/static_mode.ts` flips a flag that:
   - swaps the WS kernel client for a `StaticKernelClient` that replays
     the baked event stream
   - makes CodeMirror cells read-only
   - disables Run/Restart/Save buttons
   - replaces the workdir browser with a manifest-driven examples list
   - auto-runs all cells on file open

3. The build output (`python/python_src/rapidfem/ui/frontend/dist/`)
   is published to GitHub Pages by `.github/workflows/deploy-demo.yml`.

## Local preview

```bash
# Re-bake (only needed when examples or rapidfem itself change)
python scripts/bake_demo.py

# Build with static mode
cd python/python_src/rapidfem/ui/frontend-src
VITE_STATIC_MODE=1 npm run build

# Serve the dist/ statically
cd ../frontend
python -m http.server 8000 --directory dist
# → http://localhost:8000/
```

## Deployment: `fem.rapidpassives.org`

### One-time setup

1. **Repository settings → Pages**: set source to **"GitHub Actions"**.
   The workflow uses the official `actions/deploy-pages` flow which
   requires this mode.

2. **DNS** (at your domain registrar for `rapidpassives.org`): add a
   `CNAME` record:

   ```
   Type:  CNAME
   Name:  fem
   Value: milanofthe.github.io
   TTL:   3600 (or default)
   ```

   GitHub Pages will then automatically provision HTTPS via Let's
   Encrypt. The repo already contains
   `python/python_src/rapidfem/ui/frontend-src/static/CNAME` with
   `fem.rapidpassives.org`, which Pages reads from `dist/CNAME` after
   the build.

3. **Verify**: after the first successful deploy and DNS propagation
   (5–60 min), hit `https://fem.rapidpassives.org/`.

### Trigger a deploy

- Any push to `master` (incl. merges from feature branches).
- Or: GitHub UI → Actions → "Deploy static demo" → Run workflow.

## Operational notes

- The bake step needs the same Python deps as the live UI server
  (`rapidfem` editable + `[ui]` extra) and uses faer's LU solver
  on the CI runner (no MKL/PARDISO needed). Total bake time for the
  current 5 examples is ~20 s on `ubuntu-latest`.
- Bundle size: ~7 MB raw, ~2–3 MB transferred (GH Pages auto-gzips).
- The `static/demo/` directory is gitignored — CI regenerates it on
  every deploy. Don't check in baked artefacts.
- The kernel-WS protocol on the live server and the `StaticKernelClient`
  consume the *same* display-event shape. If you change `serialize.py`
  or `_serialize_captures_for_protocol`, re-bake.
