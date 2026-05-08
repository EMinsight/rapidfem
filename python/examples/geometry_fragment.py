"""
Stress test: Box + XYPlate, fragment, then verify named faces still resolve
to the right gmsh entities post-fragment. This is the patch-antenna scenario.
"""
import sys

import rapidfem


def main() -> int:
    g = rapidfem.Geometry()
    sub = g.box(10, 10, 1, position=(0, 0, 0))
    plate = g.xy_plate(6, 6, position=(2, 2, 1))

    # Tag faces BEFORE fragment
    sub.faces.min(axis="z").name = "ground"
    sub.faces.where(lambda c, _: abs(c[0] - 0) < 1e-9).name = "wall_x_low"
    sub.faces.where(lambda c, _: abs(c[0] - 10) < 1e-9).name = "wall_x_high"
    plate.name = "patch"
    sub.material = "fr4"

    # Fragment: embed the plate into the substrate's top face
    g.fragment(sub, plate)

    mesh_bytes, name_to_tag = g.mesh(maxh=2.0)
    g.close()

    print(f"Mesh: {len(mesh_bytes) / 1024:.1f} KB")
    print("name_to_tag:")
    for n, t in sorted(name_to_tag.items()):
        print(f"  {n!r}: tag {t}")

    # Sanity: each unique name resolved to a tag
    expected = {"ground", "wall_x_low", "wall_x_high", "patch", "fr4"}
    missing = expected - set(name_to_tag.keys())
    if missing:
        print(f"FAIL: missing names {missing}")
        return 1
    if len(set(name_to_tag.values())) != len(name_to_tag):
        print(f"FAIL: tag collisions in {name_to_tag}")
        return 1
    print("OK: all 5 names resolved to distinct tags after fragment()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
