# RapidFEM Documentation Site

Static documentation site for RapidFEM — API reference and guides. Built
with SvelteKit (`adapter-static`), deployed to `fem.rapidpassives.org/docs`
alongside the notebook demo.

This directory lives outside `python/`, so it is **never bundled into the
PyPI wheel**.

## Layout

```
docs-site/
  src/routes/+page.svelte          landing page (overview, install, quickstart)
  src/routes/[version]/api/        versioned API reference
  src/lib/components/api/          API rendering (modules, classes, functions)
  src/lib/components/layout/       header, sidebar, doc layout
  src/lib/config/rapidfem.ts       site content — features, install, modules
  scripts/build.py                 API extraction (griffe static analysis)
  static/api/                      generated API JSON (one per version)
```

## Develop

```bash
npm install
npm run extract     # extract API JSON for every git tag (needs Python deps)
npm run dev         # dev server at http://localhost:5173
```

`npm run extract` needs the Python build dependencies:

```bash
pip install -r scripts/requirements.txt
```

It extracts the public API of every released tag (`>= v0.5.0`) into
`static/api/<tag>.json` via [griffe](https://mkdocstrings.github.io/griffe/)
static analysis — RapidFEM itself does **not** need to be installed. Each tag
is checked out in a throwaway git worktree, so the working tree is untouched.
Add `--head` to also extract the current working tree as version `dev`.

## Build

```bash
npm run build       # output in build/
```

For the production deploy the site is built with `BASE_PATH=/docs` so all
links resolve under `fem.rapidpassives.org/docs`.

## Deploy

`.github/workflows/deploy-demo.yml` builds the notebook demo and this docs
site on every push to `master`, assembles them into a single GitHub Pages
artifact (demo at `/`, docs at `/docs`) and deploys it.

## Examples

The worked examples are the pre-baked notebooks of the static demo at
`fem.rapidpassives.org` — the sidebar "Examples" entry links there. The docs
site itself does not render notebooks.
