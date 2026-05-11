#!/usr/bin/env bash
# Build the SvelteKit frontend and place dist/ under
# python/python_src/rapidfem/ui/frontend/dist so that maturin develop /
# rapidfem serve can find it via importlib.resources.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$repo/python/python_src/rapidfem/ui/frontend-src"
dest="$repo/python/python_src/rapidfem/ui/frontend/dist"

if [[ ! -d "$src" ]]; then
    echo "error: frontend source not found at $src" >&2
    exit 1
fi

cd "$src"
if [[ ! -d node_modules ]]; then
    echo ">> npm ci"
    npm ci
fi
echo ">> npm run build"
npm run build

# svelte.config.js (adapter-static) will be configured in P4.7 to output
# directly into ../frontend/dist. Until then, copy from the default build dir.
if [[ -d "$src/build" ]]; then
    rm -rf "$dest"
    mkdir -p "$dest"
    cp -R "$src/build/." "$dest/"
    echo ">> Copied $src/build -> $dest"
fi

echo "Frontend built at: $dest"
