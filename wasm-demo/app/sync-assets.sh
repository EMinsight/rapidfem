#!/usr/bin/env bash
# Pull the latest WASM build + demo .msh/.toml assets into static/.
# Run this whenever you rebuild rapidfem-wasm or regenerate demos via
# wasm-demo/scripts/dump_*.py. Vite then serves them with hot-reload.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cp "$HERE/../pkg/"*.{js,wasm,d.ts,json} "$HERE/static/pkg/" 2>/dev/null || true
cp "$HERE/../web/"{wr90_straight.msh,wr90.toml,microstrip.msh,microstrip.toml,spiral.msh,spiral.toml} \
   "$HERE/static/examples/"
echo "synced WASM pkg + .msh/.toml assets into $HERE/static/"
