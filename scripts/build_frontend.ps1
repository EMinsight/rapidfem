#!/usr/bin/env pwsh
# Build the SvelteKit frontend. adapter-static writes directly into
# python/python_src/rapidfem/ui/frontend/dist via svelte.config.js so
# importlib.resources can find it.

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

Write-Host "Frontend built at: $dest"
