#!/usr/bin/env pwsh
# Build the SvelteKit frontend and place the dist/ under
# python/python_src/rapidfem/ui/frontend/dist so that maturin develop /
# rapidfem serve can find it via importlib.resources.

$ErrorActionPreference = "Stop"

$repo = Split-Path $PSScriptRoot -Parent
$src  = Join-Path $repo "python/python_src/rapidfem/ui/frontend-src"
$dest = Join-Path $repo "python/python_src/rapidfem/ui/frontend/dist"

if (-not (Test-Path $src)) {
    Write-Error "Frontend source not found at $src"
}

Push-Location $src
try {
    if (-not (Test-Path "node_modules")) {
        Write-Host ">> npm ci"
        npm ci
        if ($LASTEXITCODE -ne 0) { throw "npm ci failed" }
    }
    Write-Host ">> npm run build"
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
}
finally {
    Pop-Location
}

# svelte.config.js (adapter-static) will be configured in P4.7 to output
# directly into ../frontend/dist. Until then, copy from the default build dir.
$built = Join-Path $src "build"
if (Test-Path $built) {
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    Copy-Item -Path (Join-Path $built "*") -Destination $dest -Recurse -Force
    Write-Host ">> Copied $built -> $dest"
}

Write-Host "Frontend built at: $dest"
