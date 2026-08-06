"""Process-stack data model: `PdkLayer` + `Stack`, mirrors rapidpassives.

The `Stack` data model mirrors the schema used by ``rapidpassives`` so that
the same JSON describes geometry on both sides:

- Each `PdkLayer` carries (name, gds, datatype, z, thickness, color, type)
- `Stack.from_pdk("sky130")` returns a fully-populated stack ready for GDS
  loading and FEM extrusion
- `stack.to_dict()` round-trips to the rapidpassives `Pdk` JSON shape
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from rapidfem.geometry import Geometry, GeoObject


LayerType = Literal["metal", "via", "poly", "diffusion", "substrate", "oxide", "other"]


@dataclass
class PdkLayer:
    """One physical layer in a process stack.

    Mirrors ``rapidpassives.web.lib.stack.pdk.PdkLayer`` 1:1 so the same JSON
    describes a stack on both rapidfem (FEM-side) and rapidpassives (viewer-side).

    Coordinate convention: z is the BOTTOM of the layer (lower face). The top
    is at z + thickness. All distances in meters (rapidpassives uses microns;
    converters below help round-trip).
    """
    name: str
    gds: int
    datatype: int
    z: float                # bottom z, in meters
    thickness: float        # in meters
    color: str = "#888"
    type: LayerType = "metal"
    # Material defaults (FEM-relevant; rapidpassives ignores these)
    er: float = 1.0
    ur: float = 1.0
    tand: float = 0.0
    sigma: float = 0.0       # bulk conductivity (S/m); 0 ⇒ treat as PEC for metals

    @property
    def z_top(self) -> float:
        return self.z + self.thickness

    @property
    def gds_key(self) -> tuple[int, int]:
        return (self.gds, self.datatype)


@dataclass
class Stack:
    """A complete process stack: ordered physical layers + substrate properties.

    Use ``Stack.sky130()`` / ``Stack.sg13g2()`` for built-in presets, or
    construct manually from a list of `PdkLayer`s.
    """
    name: str
    layers: list[PdkLayer]
    # Substrate slab below the lowest layer (silicon wafer).
    substrate_thickness: float = 300e-6   # m
    substrate_er: float = 11.9
    substrate_sigma: float = 10.0          # S/m (lossy silicon)
    # Bulk dielectric between metals (often modeled as a single effective er)
    oxide_er: float = 4.2
    oxide_tand: float = 0.0

    # ── Construction helpers ───────────────────────────────────────────────

    def __post_init__(self):
        # Sort layers bottom-to-top by z for deterministic iteration
        self.layers = sorted(self.layers, key=lambda l: (l.z, l.thickness))

    @staticmethod
    def from_pdk(name: str) -> "Stack":
        """Convenience: dispatch by lowercased PDK name."""
        normalized = name.lower().replace("-", "").replace("_", "")
        if normalized in ("sky130", "skywater130"):
            return Stack.sky130()
        if normalized in ("sg13g2", "ihpsg13g2"):
            return Stack.sg13g2()
        raise ValueError(f"unknown PDK {name!r}; available: sky130, sg13g2")

    @staticmethod
    def sky130() -> "Stack":
        """SkyWater SKY130 open-source PDK process stack.

        Layer numbering and z-positions match the rapidpassives `pdk.ts` file
        (z=0 at the bottom of li1, polysilicon below at z=-0.18 μm).
        """
        um = 1e-6
        layers = [
            # name        gds  dt   z(μm)  t(μm)   color    type      er    sigma
            PdkLayer("poly",   66, 20, -0.18 * um, 0.18 * um, "#c4725e", "poly",     er=4.2),
            PdkLayer("licon1", 66, 44, -0.10 * um, 0.10 * um, "#5a5a62", "via",      sigma=4.1e7),
            PdkLayer("li1",    67, 20,  0.00 * um, 0.10 * um, "#7b5e8a", "metal",    sigma=4.1e7),
            PdkLayer("mcon",   67, 44,  0.10 * um, 0.27 * um, "#5a5a62", "via",      sigma=4.1e7),
            PdkLayer("met1",   68, 20,  0.37 * um, 0.36 * um, "#6bbf8a", "metal",    sigma=4.1e7),
            PdkLayer("via",    68, 44,  0.73 * um, 0.27 * um, "#5a5a62", "via",      sigma=4.1e7),
            PdkLayer("met2",   69, 20,  1.00 * um, 0.36 * um, "#4a9ec2", "metal",    sigma=4.1e7),
            PdkLayer("via2",   69, 44,  1.36 * um, 0.42 * um, "#6e6e78", "via",      sigma=4.1e7),
            PdkLayer("met3",   70, 20,  1.78 * um, 0.845 * um, "#5aad78", "metal",   sigma=4.1e7),
            PdkLayer("via3",   70, 44,  2.625 * um, 0.39 * um, "#6e6e78", "via",     sigma=4.1e7),
            PdkLayer("met4",   71, 20,  3.015 * um, 0.845 * um, "#d9513c", "metal",  sigma=4.1e7),
            PdkLayer("via4",   71, 44,  3.86 * um, 0.505 * um, "#7a7a84", "via",     sigma=4.1e7),
            PdkLayer("met5",   72, 20,  4.365 * um, 1.26 * um, "#e8944a", "metal",   sigma=4.1e7),
        ]
        return Stack(
            name="SKY130", layers=layers,
            substrate_thickness=300 * um,
            substrate_er=11.9, substrate_sigma=10.0,
            oxide_er=4.2, oxide_tand=0.0,
        )

    @staticmethod
    def sg13g2() -> "Stack":
        """IHP SG13G2 open-source 130 nm SiGe BiCMOS PDK process stack."""
        um = 1e-6
        layers = [
            PdkLayer("GatPoly", 5,  0, -0.16 * um, 0.16 * um, "#c4725e", "poly",   er=4.2),
            PdkLayer("Metal1",  8,  0,  0.00 * um, 0.42 * um, "#7b5e8a", "metal",  sigma=4.1e7),
            PdkLayer("Via1",    19, 0,  0.42 * um, 0.54 * um, "#5a5a62", "via",    sigma=4.1e7),
            PdkLayer("Metal2",  10, 0,  0.96 * um, 0.49 * um, "#6bbf8a", "metal",  sigma=4.1e7),
            PdkLayer("Via2",    29, 0,  1.45 * um, 0.54 * um, "#6e6e78", "via",    sigma=4.1e7),
            PdkLayer("Metal3",  30, 0,  1.99 * um, 0.49 * um, "#4a9ec2", "metal",  sigma=4.1e7),
            PdkLayer("Via3",    49, 0,  2.48 * um, 0.54 * um, "#6e6e78", "via",    sigma=4.1e7),
            PdkLayer("Metal4",  50, 0,  3.02 * um, 0.49 * um, "#5aad78", "metal",  sigma=4.1e7),
            PdkLayer("Via4",    66, 0,  3.51 * um, 0.54 * um, "#7a7a84", "via",    sigma=4.1e7),
            PdkLayer("Metal5",  67, 0,  4.05 * um, 0.49 * um, "#d9513c", "metal",  sigma=4.1e7),
            PdkLayer("TopVia1", 125, 0, 4.54 * um, 0.85 * um, "#7a7a84", "via",    sigma=4.1e7),
            PdkLayer("TopMetal1", 126, 0, 5.39 * um, 2.0 * um, "#e8944a", "metal", sigma=4.1e7),
            PdkLayer("TopVia2", 133, 0, 7.39 * um, 2.85 * um, "#7a7a84", "via",    sigma=4.1e7),
            PdkLayer("TopMetal2", 134, 0, 10.24 * um, 3.0 * um, "#e8944a", "metal",sigma=4.1e7),
        ]
        return Stack(
            name="SG13G2", layers=layers,
            substrate_thickness=200 * um,
            substrate_er=11.9, substrate_sigma=20.0,   # SG13G2 is moderately resistive
            oxide_er=4.1, oxide_tand=0.0,
        )

    # ── Lookups ────────────────────────────────────────────────────────────

    def by_name(self, name: str) -> PdkLayer:
        for l in self.layers:
            if l.name == name:
                return l
        raise KeyError(f"layer {name!r} not in stack {self.name!r}; available: "
                       f"{[l.name for l in self.layers]}")

    def by_gds(self, gds: int, datatype: int = 0) -> PdkLayer | None:
        """Find the layer matching a GDS (number, datatype) tuple. None if absent."""
        for l in self.layers:
            if l.gds == gds and l.datatype == datatype:
                return l
        return None

    def metals(self) -> list[PdkLayer]:
        return [l for l in self.layers if l.type == "metal"]

    def vias(self) -> list[PdkLayer]:
        return [l for l in self.layers if l.type == "via"]

    @property
    def top_z(self) -> float:
        return max((l.z_top for l in self.layers), default=0.0)

    @property
    def bottom_z(self) -> float:
        """z of the highest substrate top (where active devices sit). Substrate
        slab itself sits below this at [bottom_z - substrate_thickness, bottom_z]."""
        return min((l.z for l in self.layers), default=0.0)

    # ── JSON interop with rapidpassives ────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to the rapidpassives `Pdk` JSON shape (lengths in microns)."""
        um = 1e-6
        return {
            "id": self.name.lower(),
            "name": self.name,
            "description": f"rapidfem stack: {self.name}",
            "substrate": {
                "thickness_um": self.substrate_thickness / um,
                "er": self.substrate_er,
                "sigma": self.substrate_sigma,
            },
            "oxide": {"er": self.oxide_er, "tand": self.oxide_tand},
            "layers": [
                {
                    "name": l.name, "gds": l.gds, "datatype": l.datatype,
                    "z_um": l.z / um, "thickness_um": l.thickness / um,
                    "color": l.color, "type": l.type,
                    "er": l.er, "ur": l.ur, "tand": l.tand, "sigma": l.sigma,
                }
                for l in self.layers
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "Stack":
        um = 1e-6
        sub = d.get("substrate", {})
        ox = d.get("oxide", {})
        layers = [
            PdkLayer(
                name=l["name"], gds=l["gds"], datatype=l["datatype"],
                z=l["z_um"] * um, thickness=l["thickness_um"] * um,
                color=l.get("color", "#888"), type=l.get("type", "metal"),
                er=l.get("er", 1.0), ur=l.get("ur", 1.0),
                tand=l.get("tand", 0.0), sigma=l.get("sigma", 0.0),
            )
            for l in d["layers"]
        ]
        return Stack(
            name=d["name"], layers=layers,
            substrate_thickness=sub.get("thickness_um", 300) * um,
            substrate_er=sub.get("er", 11.9), substrate_sigma=sub.get("sigma", 10.0),
            oxide_er=ox.get("er", 4.2), oxide_tand=ox.get("tand", 0.0),
        )

    # ── Geometry helpers ───────────────────────────────────────────────────

    def create_substrate(
        self,
        g: "Geometry",
        footprint: tuple[float, float],
        center: bool = True,
        z_substrate_top: float | None = None,
        fragment_existing: bool = True,
    ) -> dict[str, "GeoObject"]:
        """Instantiate the silicon substrate slab and a bulk-oxide slab spanning
        from the substrate top to the stack's top.

        Each block is created with a fully-instantiated ``rf.Dielectric``
        derived from the stack constants, drop the returned objects straight
        into the Problem API, no extra material wiring needed.

        Returns a dict of named GeoObjects (`substrate`, `oxide`). If
        ``fragment_existing=True`` (the default) and the geometry already
        contains 3D primitives (e.g. metal traces from `Geometry.from_gds`),
        they are fragmented into the new oxide slab so the resulting mesh is
        conformal at every interface.
        """
        # Local import, avoids a circular at module load (rapidfem.__init__
        # imports rapidfem.rfic).
        from rapidfem.materials import Dielectric

        wx, wy = footprint
        x0 = -wx / 2 if center else 0.0
        y0 = -wy / 2 if center else 0.0
        z_top = z_substrate_top if z_substrate_top is not None else self.bottom_z

        # Snapshot existing 3D objects BEFORE adding substrate/oxide
        existing_3d = [o for o in g._objects if o.dim == 3]

        silicon = Dielectric(er=self.substrate_er, conductivity=self.substrate_sigma)
        sio2 = Dielectric(er=self.oxide_er, tand=self.oxide_tand)

        sub = g.box(wx, wy, self.substrate_thickness,
                    position=(x0, y0, z_top - self.substrate_thickness),
                    material=silicon)
        sub.name = "substrate"

        oxide_height = self.top_z - z_top
        ox = None
        if oxide_height > 0:
            ox = g.box(wx, wy, oxide_height, position=(x0, y0, z_top),
                       material=sio2)
            ox.name = "oxide"

        # Fragment with all pre-existing 3D primitives so interfaces are conformal.
        # Critical: do this in ONE call. Two sequential fragment ops carve the
        # second target against the same tools but leave the first one in a
        # half-resolved state, re-resolution by (cog, bbox) then misattributes
        # the first volume's name to the wrong sub-piece (#64).
        if fragment_existing and existing_3d:
            others = existing_3d + ([ox] if ox is not None else [])
            g.fragment(sub, *others)

        return {"substrate": sub} | ({"oxide": ox} if ox is not None else {})


__all__ = ["Stack", "PdkLayer", "LayerType"]
