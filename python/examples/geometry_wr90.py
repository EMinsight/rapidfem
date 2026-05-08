"""
WR-90 straight waveguide built entirely with the new Geometry builder API.

No raw gmsh code, no bounding-box matching — just primitives + selectors + names.
Compares to the existing TOML+mesh-file flow: should produce equivalent S-parameters.
"""
import os
import sys

import numpy as np
import rapidfem


def main() -> int:
    a = 22.86e-3   # WR-90 broad wall
    b = 10.16e-3   # narrow wall
    L = 30e-3      # length

    g = rapidfem.Geometry()
    box = g.box(a, b, L, position=(0, 0, 0))

    # Tag walls + ports by face selectors
    box.faces.where(lambda c, _: abs(c[0] - 0) < 1e-9).name = "pec_wall"
    box.faces.where(lambda c, _: abs(c[0] - a) < 1e-9).name = "pec_wall"
    box.faces.where(lambda c, _: abs(c[1] - 0) < 1e-9).name = "pec_wall"
    box.faces.where(lambda c, _: abs(c[1] - b) < 1e-9).name = "pec_wall"
    box.faces.min(axis="z").name = "port1"
    box.faces.max(axis="z").name = "port2"
    box.material = "air"

    mesh_bytes, name_to_tag = g.mesh(maxh=3e-3)
    print(f"Mesh: {len(mesh_bytes)/1024:.1f} KB, tags: {name_to_tag}")
    g.close()

    # Build the Simulation TOML using the resolved tag numbers.
    # (SimulationBuilder will replace this in task #48.)
    pec_tag = name_to_tag["pec_wall"]
    p1_tag = name_to_tag["port1"]
    p2_tag = name_to_tag["port2"]
    config_toml = f"""
[mesh]
file = "(in-memory)"

[frequency]
range = [9.0e9, 11.0e9, 11]

[[ports]]
type = "rectangular"
tag = {p1_tag}
width = {a}
height = {b}

[[ports]]
type = "rectangular"
tag = {p2_tag}
width = {a}
height = {b}

[pec]
tags = [{pec_tag}]
"""

    sim = rapidfem.Simulation.from_bytes(mesh_bytes, config_toml)
    print(f"Sim: {sim.n_tets} tets, {sim.n_dofs} DOFs, {sim.n_driven_ports} driven ports")
    result = sim.run_sweep()

    s11_max = float(np.abs(result.sparams[:, 0, 0]).max())
    s21_min = float(np.abs(result.sparams[:, 1, 0]).min())
    s21_max = float(np.abs(result.sparams[:, 1, 0]).max())
    print(f"max |S11| = {s11_max:.5f}  (expected << 1)")
    print(f"|S21| range = [{s21_min:.5f}, {s21_max:.5f}]  (expected ~1)")

    if s11_max < 0.01 and abs(s21_min - 1.0) < 0.01 and abs(s21_max - 1.0) < 0.01:
        print("OK")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
