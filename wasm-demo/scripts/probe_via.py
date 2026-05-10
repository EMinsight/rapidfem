"""Inspect rp_spiral.msh: what physical groups do its surface entities belong to?"""
import re

with open(r"C:\Repositories\rapidfem\wasm-demo\app\static\examples\rp_spiral.msh") as f:
    txt = f.read()

m = re.search(r"\$PhysicalNames\n(\d+)\n(.*?)\$EndPhysicalNames", txt, re.DOTALL)
phys = {}
for line in m.group(2).strip().split("\n"):
    parts = line.split(" ", 2)
    phys[int(parts[1])] = parts[2].strip('"')
print("physical groups:", phys)

m = re.search(r"\$Entities\n(.*?)\$EndEntities", txt, re.DOTALL)
elines = m.group(1).strip().split("\n")
header = list(map(int, elines[0].split()))
nPts, nCurves, nSurfs, nVols = header
print(f"counts: pts={nPts} curves={nCurves} surfs={nSurfs} vols={nVols}")

i = 1 + nPts + nCurves
print("--- surfaces with their physical groups ---")
named = 0
for s in range(nSurfs):
    parts = elines[i + s].split()
    tag = int(parts[0])
    nphys = int(parts[7])
    phys_list = [int(parts[8 + k]) for k in range(nphys)]
    names = [phys.get(p, "?") for p in phys_list]
    if names:
        named += 1
        print(f"  surf {tag}: {names}")
print(f"total named surfaces: {named}")

# Count tris per phys-group via element blocks
m = re.search(r"\$Elements\n(\d+) (\d+) (\d+) (\d+)\n(.*?)\$EndElements", txt, re.DOTALL)
nblocks = int(m.group(1))
elines = m.group(5).split("\n")
i = 0
# Build entity_tag -> first physical name from above iteration
ent2phys = {}
ei = 1 + nPts + nCurves
m2 = re.search(r"\$Entities\n(.*?)\$EndEntities", txt, re.DOTALL)
elines2 = m2.group(1).strip().split("\n")
for s in range(nSurfs):
    parts = elines2[ei + s].split()
    tag = int(parts[0])
    nphys = int(parts[7])
    phys_list = [int(parts[8 + k]) for k in range(nphys)]
    if phys_list:
        ent2phys[tag] = phys.get(phys_list[0], "?")

print("--- tri counts per entity (only those in physical groups) ---")
from collections import defaultdict
phys_tris = defaultdict(int)
for b in range(nblocks):
    bh = list(map(int, elines[i].split()))
    if bh[0] == 2 and bh[2] == 2 and bh[1] in ent2phys:
        phys_tris[ent2phys[bh[1]]] += bh[3]
    i += 1 + bh[3]
for k, v in sorted(phys_tris.items(), key=lambda kv: -kv[1]):
    print(f"  {k}: {v} tris")
