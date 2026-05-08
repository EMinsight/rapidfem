"""WR-90 driven entirely by Geometry + SimulationBuilder. No TOML strings, no integer tags."""
import sys
import numpy as np
import rapidfem


def main() -> int:
    a, b, L = 22.86e-3, 10.16e-3, 30e-3

    # 1) Geometry
    g = rapidfem.Geometry()
    box = g.box(a, b, L)
    box.faces.where(lambda c, _: abs(c[0] - 0) < 1e-9).name = "pec_wall"
    box.faces.where(lambda c, _: abs(c[0] - a) < 1e-9).name = "pec_wall"
    box.faces.where(lambda c, _: abs(c[1] - 0) < 1e-9).name = "pec_wall"
    box.faces.where(lambda c, _: abs(c[1] - b) < 1e-9).name = "pec_wall"
    box.faces.min(axis="z").name = "port1"
    box.faces.max(axis="z").name = "port2"
    box.material = "air"

    # 2) Simulation built fluently
    sim = (
        rapidfem.SimulationBuilder()
        .from_geometry(g, maxh=3e-3)
        .frequencies(np.linspace(9e9, 11e9, 11))
        .pec("pec_wall")
        .rect_waveguide("port1", mode=(1, 0), width=a, height=b)
        .rect_waveguide("port2", mode=(1, 0), width=a, height=b)
        .material("air", er=1.0)
        .build()
    )
    g.close()

    result = sim.run_sweep()
    s11_max = float(np.abs(result.sparams[:, 0, 0]).max())
    s21_min = float(np.abs(result.sparams[:, 1, 0]).min())
    s21_max = float(np.abs(result.sparams[:, 1, 0]).max())
    print(f"max |S11| = {s11_max:.5f}, |S21| range = [{s21_min:.5f}, {s21_max:.5f}]")

    if s11_max < 0.01 and abs(s21_min - 1.0) < 0.01 and abs(s21_max - 1.0) < 0.01:
        print("OK")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
