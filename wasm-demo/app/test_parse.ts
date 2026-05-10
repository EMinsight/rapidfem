import { parse_msh } from './src/lib/msh.ts';
import { readFileSync } from 'fs';
const txt = readFileSync('static/examples/rp_spiral.msh', 'utf-8');
const m = parse_msh(txt);
console.log('phys_names:', [...m.phys_names.entries()]);
console.log('phys_dim:', [...m.phys_dim.entries()]);
const counts = new Map<number, number>();
for (const t of m.tri_phys) counts.set(t, (counts.get(t) ?? 0) + 1);
console.log('tri_phys counts:', [...counts.entries()].sort((a, b) => b[1] - a[1]));
