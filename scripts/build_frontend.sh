#!/usr/bin/env bash
# Build the SvelteKit frontend. adapter-static writes directly into
# python/python_src/rapidfem/ui/frontend/dist via svelte.config.js so
# importlib.resources can find it.

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

echo "Frontend built at: $dest"
