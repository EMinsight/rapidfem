// Ambient type declarations for the RapidFEM frontend.
//
// No top-level import/export here — that would make this a module and the
// `declare module` below a module *augmentation* rather than a global
// ambient declaration.

// `plotly.js-dist-min` ships no bundled type declarations; the plotting
// panels (ResultsPanel, TimeSeriesPanel) use it dynamically imported.
declare module 'plotly.js-dist-min';
