"""Composite RF structure builders for :class:`rapidfem.geometry.Geometry`.

Module-level functions (in the spirit of :meth:`Geometry.from_gds`) that
compose the primitive builders, plus optionally the standard physics, into
common macroscopic EM setups: coaxial lines, microstrip lines, coplanar
waveguide, ... Each takes a :class:`Geometry` as its first argument, builds
into it, and returns a small dataclass holding the created objects and the
canonical port faces.

These are **not** the PDK-stack-driven RFIC helpers in
:mod:`rapidfem.rfic` (those take a :class:`rapidfem.rfic.Stack` and named
metal layers at micrometre scale). The builders here stand alone: they
create their own substrate / air / conductor geometry from physical
dimensions.

Geometry only by default. Pass ``add_ports=True`` to also attach the
canonical ports plus the enclosing PEC so the structure is immediately
solvable; otherwise use the returned port faces to wire your own physics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .materials import Air, Dielectric
from .physics import CoaxPort, PEC

if TYPE_CHECKING:
    from .geometry import EntityCollection, GeoObject, Geometry


# Map the axis label a builder is laid out along to its unit direction. Used
# both for the cylinder sweep direction and for selecting the end-cap faces.
_AXIS_VEC = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}


@dataclass
class CoaxLine:
    """Result of :func:`coax`.

    Attributes
    ----------
    dielectric : GeoObject
        the coaxial body (outer conductor radius down to the inner-conductor
        surface); carries the fill material. Its end-cap faces are the ports.
    port_a : EntityCollection
        the end cap at the base of the line (minimum along the build axis)
    port_b : EntityCollection
        the end cap at the far end (maximum along the build axis)
    ports : list
        the two :class:`rapidfem.CoaxPort` objects, populated only when
        :func:`coax` was called with ``add_ports=True`` (else empty)
    """

    dielectric: "GeoObject"
    port_a: "EntityCollection"
    port_b: "EntityCollection"
    ports: list = field(default_factory=list)


def coax(g: "Geometry", *,
         ri: float, ro: float, length: float,
         origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
         axis: str = "z",
         er: float = 1.0,
         material=None,
         add_ports: bool = False,
         power: float = 1.0) -> CoaxLine:
    """build a straight coaxial line: a fill cylinder of outer radius ``ro``
    with the inner conductor (radius ``ri``) removed as a PEC core.

    The annular region between ``ri`` and ``ro`` carries the fill material
    (air by default, or a dielectric when ``er`` is set). The two end caps
    are the coaxial ports.


    Example
    -------
    A 20 mm matched 50 ohm air line, ready to solve:

    .. code-block:: python

        from rapidfem import structures as st
        cx = st.coax(g, ri=1.5e-3, ro=3.45e-3, length=20e-3, add_ports=True)

    Geometry only, wiring your own ports off the returned faces:

    .. code-block:: python

        cx = st.coax(g, ri=1.5e-3, ro=3.45e-3, length=20e-3)
        rf.CoaxPort(cx.port_a, ri=1.5e-3, ro=3.45e-3)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    ri, ro : float
        inner and outer conductor radii in metres (``ri < ro``)
    length : float
        line length in metres along ``axis``
    origin : tuple[float, float, float]
        base-cap centre (defaults to the origin)
    axis : str
        build direction, one of ``"x"`` / ``"y"`` / ``"z"`` (defaults to z)
    er : float
        relative permittivity of the fill (defaults to 1, i.e. air); ignored
        when ``material`` is given
    material : rapidfem.Material, optional
        explicit fill material; overrides ``er``
    add_ports : bool
        when True, attach a :class:`rapidfem.CoaxPort` at each end and PEC on
        every remaining (inner-conductor + shield) face
    power : float
        port excitation power in watts (only used when ``add_ports``)

    Returns
    -------
    CoaxLine
        the built coaxial line and its port faces

    Raises
    ------
    ValueError
        if ``ri >= ro`` or ``axis`` is not one of x / y / z
    """
    if ri >= ro:
        raise ValueError(f"coax: ri ({ri}) must be < ro ({ro})")
    if axis not in _AXIS_VEC:
        raise ValueError(f"coax: axis must be 'x', 'y' or 'z', got {axis!r}")
    av = _AXIS_VEC[axis]
    fill = material if material is not None else (
        Air() if er == 1.0 else Dielectric(er=er))

    # Outer fill cylinder with the inner-conductor cylinder fragmented out, so
    # the inner-conductor surface exists as a face we can mark PEC. The core's
    # material is irrelevant (it is walled off by PEC), so it shares the fill.
    outer = g.cylinder(ro, length, position=origin, axis=av, material=fill)
    inner = g.cylinder(ri, length, position=origin, axis=av, material=fill)
    g.fragment(outer, inner)

    port_a = outer.faces.min(axis=axis)
    port_b = outer.faces.max(axis=axis)
    line = CoaxLine(dielectric=outer, port_a=port_a, port_b=port_b)

    if add_ports:
        far = tuple(origin[i] + av[i] * length for i in range(3))
        p0 = CoaxPort(port_a, ri=ri, ro=ro, origin=origin, z_axis=av,
                      er=er, power=power)
        p1 = CoaxPort(port_b, ri=ri, ro=ro, origin=far, z_axis=av,
                      er=er, power=power)
        # Everything left (inner-conductor surface + outer shield) is PEC.
        PEC(*outer.faces.unassigned)
        line.ports = [p0, p1]

    return line
