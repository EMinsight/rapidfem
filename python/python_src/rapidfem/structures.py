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
from .physics import ABC, CoaxPort, PEC, WavePort

if TYPE_CHECKING:
    from .geometry import EntityCollection, GeoObject, Geometry


# Map the axis label a builder is laid out along to its unit direction. Used
# both for the cylinder sweep direction and for selecting the end-cap faces.
_AXIS_VEC = {
    "x": (1.0, 0.0, 0.0),
    "y": (0.0, 1.0, 0.0),
    "z": (0.0, 0.0, 1.0),
}

# Microstrip substrate ports: a single-element-thick substrate slab is too
# coarse for the vector wave-port eigensolve to resolve the inhomogeneous
# quasi-TEM mode, so the substrate is meshed at this fraction of its
# thickness by default. Matches the fd_microstrip_line.py example.
_SUBSTRATE_MESH_DIVISIONS = 3


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


@dataclass
class MicrostripLine:
    """Result of :func:`microstrip`.

    Attributes
    ----------
    substrate : GeoObject
        the dielectric substrate slab
    air : GeoObject
        the air region above the substrate
    trace : GeoObject
        the signal trace (a thin plate on top of the substrate)
    ground : EntityCollection
        the substrate's bottom face (the ground plane; PEC when
        ``add_ports``)
    port_a : tuple[EntityCollection, EntityCollection]
        the (substrate, air) cross-section faces at the line's near end
    port_b : tuple[EntityCollection, EntityCollection]
        the (substrate, air) cross-section faces at the line's far end
    pec : object or None
        the :class:`rapidfem.PEC` object covering trace + ground when
        ``add_ports`` was set (else None)
    ports : list
        the two :class:`rapidfem.WavePort` objects when ``add_ports`` was
        set (else empty)
    """

    substrate: "GeoObject"
    air: "GeoObject"
    trace: "GeoObject"
    ground: "EntityCollection"
    port_a: "tuple[EntityCollection, EntityCollection]"
    port_b: "tuple[EntityCollection, EntityCollection]"
    pec: object = None
    ports: list = field(default_factory=list)


def microstrip(g: "Geometry", *,
               line_w: float, line_l: float,
               sub_w: float, sub_h: float, air_h: float,
               er: float, tand: float = 0.0,
               origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
               sub_maxh: float | None = None,
               add_ports: bool = False,
               f0: float | None = None,
               power: float = 1.0) -> MicrostripLine:
    """build a microstrip line: a signal trace on a dielectric substrate
    over a ground plane, in an air region.

    Layout convention (fixed): the line propagates along **+y**, its width
    runs along **x**, and the substrate / air stack rises along **+z**. The
    substrate is centred on x = 0 at ``origin``; the trace sits on top of
    the substrate, centred over it.

    With ``add_ports`` the canonical full-vector :class:`rapidfem.WavePort`
    is placed on the substrate-plus-air cross-section at each end (which
    de-embeds the inhomogeneous quasi-TEM mode), the trace and ground plane
    are tied to one PEC, and a first-order :class:`rapidfem.ABC` opens the
    lateral x-walls and the air top. The wave-port eigensolve needs the band
    centre, so ``f0`` is required in that case.


    Example
    -------
    A 50 ohm line on 0.508 mm RO4003C, solvable around 3 GHz:

    .. code-block:: python

        from rapidfem import structures as st
        ms = st.microstrip(g, line_w=1.13e-3, line_l=30e-3,
                           sub_w=20e-3, sub_h=0.508e-3, air_h=10e-3,
                           er=3.55, tand=0.0027,
                           add_ports=True, f0=3.0e9)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    line_w, line_l : float
        trace width (along x) and length (along y) in metres
    sub_w : float
        substrate width along x in metres
    sub_h : float
        substrate thickness along z in metres
    air_h : float
        air-region height above the substrate along z in metres
    er : float
        substrate relative permittivity
    tand : float
        substrate loss tangent (defaults to 0)
    origin : tuple[float, float, float]
        the substrate's lower corner reference; the substrate spans
        ``x in [-sub_w/2, sub_w/2]`` about it (defaults to the origin)
    sub_maxh : float, optional
        substrate mesh size; defaults to ``sub_h / 3`` so the wave-port
        eigensolve resolves the cross-section
    add_ports : bool
        when True, attach the two wave ports, the trace + ground PEC, and
        the open-wall ABC
    f0 : float, optional
        band-centre frequency in Hz for the wave-port phase reference;
        required when ``add_ports`` is True
    power : float
        port excitation power in watts (only used when ``add_ports``)

    Returns
    -------
    MicrostripLine
        the built line, its conductors, and its port faces

    Raises
    ------
    ValueError
        if ``add_ports`` is True but ``f0`` was not given
    """
    if add_ports and f0 is None:
        raise ValueError(
            "microstrip: add_ports=True needs f0 (band-centre Hz) for the "
            "wave-port phase reference")

    ox, oy, oz = origin
    eff_sub_maxh = sub_maxh if sub_maxh is not None else sub_h / _SUBSTRATE_MESH_DIVISIONS
    fr4 = Dielectric(er=er, tand=tand, maxh=eff_sub_maxh)

    sub = g.box(sub_w, line_l, sub_h, position=(ox - sub_w / 2, oy, oz),
                material=fr4)
    air = g.box(sub_w, line_l, air_h, position=(ox - sub_w / 2, oy, oz + sub_h),
                material=Air())
    trace = g.xy_plate(line_w, line_l, position=(ox - line_w / 2, oy, oz + sub_h))

    g.fragment(sub, air, trace)

    ground = sub.faces.min(axis="z")
    port_a = (sub.faces.min(axis="y"), air.faces.min(axis="y"))
    port_b = (sub.faces.max(axis="y"), air.faces.max(axis="y"))
    line = MicrostripLine(substrate=sub, air=air, trace=trace, ground=ground,
                          port_a=port_a, port_b=port_b)

    if add_ports:
        # Trace + ground plane on one PEC so the wave-port eigensolve can mark
        # the conductor nodes via pec=[strip].
        strip = PEC(trace, ground)
        p0 = WavePort(port_a[0], port_a[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        p1 = WavePort(port_b[0], port_b[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        # Open the enclosure: ABC on the lateral x-walls (substrate + air) and
        # the air top. The y-extreme faces are the ports, so they are excluded.
        ABC(sub.faces.min(axis="x"), sub.faces.max(axis="x"),
            air.faces.min(axis="x"), air.faces.max(axis="x"),
            air.faces.max(axis="z"))
        line.pec = strip
        line.ports = [p0, p1]

    return line


@dataclass
class CpwLine:
    """Result of :func:`cpw`.

    Attributes
    ----------
    substrate, air : GeoObject
        the dielectric substrate and the air region above it
    signal : GeoObject
        the centre signal trace
    ground_left, ground_right : GeoObject
        the two coplanar ground strips flanking the signal
    port_a, port_b : tuple[EntityCollection, EntityCollection]
        the (substrate, air) cross-section faces at each end
    pec : object or None
        the PEC over all conductors when ``add_ports`` (else None)
    ports : list
        the two wave ports when ``add_ports`` (else empty)
    """

    substrate: "GeoObject"
    air: "GeoObject"
    signal: "GeoObject"
    ground_left: "GeoObject"
    ground_right: "GeoObject"
    port_a: "tuple[EntityCollection, EntityCollection]"
    port_b: "tuple[EntityCollection, EntityCollection]"
    pec: object = None
    ports: list = field(default_factory=list)


def cpw(g: "Geometry", *,
        signal_w: float, gap: float, line_l: float,
        sub_w: float, sub_h: float, air_h: float,
        er: float, tand: float = 0.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sub_maxh: float | None = None,
        backside_ground: bool = False,
        add_ports: bool = False,
        f0: float | None = None,
        power: float = 1.0) -> CpwLine:
    """build a coplanar waveguide: a centre signal trace flanked by two
    coplanar ground strips across a ``gap``, all on top of a substrate.

    Same fixed layout convention as :func:`microstrip` (propagation +y,
    width +x, stack +z). The signal is centred on x = 0; each ground strip
    runs from the gap edge out to the substrate edge. Pass
    ``backside_ground=True`` for conductor-backed CPW (adds the substrate
    bottom face to the PEC).

    With ``add_ports`` a full-vector :class:`rapidfem.WavePort` is placed on
    the cross-section at each end (``f0`` required), all three conductors
    ride on one PEC, and an ABC opens the air top.


    Example
    -------
    .. code-block:: python

        from rapidfem import structures as st
        cw = st.cpw(g, signal_w=0.4e-3, gap=0.2e-3, line_l=20e-3,
                    sub_w=10e-3, sub_h=0.635e-3, air_h=6e-3,
                    er=9.9, add_ports=True, f0=10e9)


    Parameters
    ----------
    g : Geometry
        geometry to build into
    signal_w : float
        signal-trace width along x in metres
    gap : float
        gap between the signal and each ground strip in metres
    line_l : float
        line length along y in metres
    sub_w : float
        substrate width along x in metres
    sub_h : float
        substrate thickness along z in metres
    air_h : float
        air-region height above the substrate along z in metres
    er : float
        substrate relative permittivity
    tand : float
        substrate loss tangent (defaults to 0)
    origin : tuple[float, float, float]
        substrate lower-corner reference; substrate spans x about it
    sub_maxh : float, optional
        substrate mesh size (defaults to ``sub_h / 3``)
    backside_ground : bool
        add the substrate bottom face to the PEC (conductor-backed CPW)
    add_ports : bool
        attach the two wave ports, the conductor PEC, and the ABC top
    f0 : float, optional
        band-centre frequency in Hz, required when ``add_ports``
    power : float
        port excitation power in watts (only when ``add_ports``)

    Returns
    -------
    CpwLine
        the built CPW and its port faces

    Raises
    ------
    ValueError
        if the ground strips would have non-positive width, or
        ``add_ports`` is set without ``f0``
    """
    ground_w = sub_w / 2 - signal_w / 2 - gap
    if ground_w <= 0:
        raise ValueError(
            f"cpw: signal_w/2 + gap ({signal_w / 2 + gap}) must be < sub_w/2 "
            f"({sub_w / 2}); ground strips have width {ground_w}")
    if add_ports and f0 is None:
        raise ValueError("cpw: add_ports=True needs f0 (band-centre Hz)")

    ox, oy, oz = origin
    eff_sub_maxh = sub_maxh if sub_maxh is not None else sub_h / _SUBSTRATE_MESH_DIVISIONS
    diel = Dielectric(er=er, tand=tand, maxh=eff_sub_maxh)

    sub = g.box(sub_w, line_l, sub_h, position=(ox - sub_w / 2, oy, oz),
                material=diel)
    air = g.box(sub_w, line_l, air_h, position=(ox - sub_w / 2, oy, oz + sub_h),
                material=Air())

    top_z = oz + sub_h
    signal = g.xy_plate(signal_w, line_l, position=(ox - signal_w / 2, oy, top_z))
    # Left ground: from the substrate's left edge to the left gap edge.
    gl_x0 = ox - sub_w / 2
    ground_left = g.xy_plate(ground_w, line_l, position=(gl_x0, oy, top_z))
    # Right ground: from the right gap edge to the substrate's right edge.
    gr_x0 = ox + signal_w / 2 + gap
    ground_right = g.xy_plate(ground_w, line_l, position=(gr_x0, oy, top_z))

    g.fragment(sub, air, signal, ground_left, ground_right)

    port_a = (sub.faces.min(axis="y"), air.faces.min(axis="y"))
    port_b = (sub.faces.max(axis="y"), air.faces.max(axis="y"))
    line = CpwLine(substrate=sub, air=air, signal=signal,
                   ground_left=ground_left, ground_right=ground_right,
                   port_a=port_a, port_b=port_b)

    if add_ports:
        conductors = [signal, ground_left, ground_right]
        if backside_ground:
            conductors.append(sub.faces.min(axis="z"))
        strip = PEC(*conductors)
        p0 = WavePort(port_a[0], port_a[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        p1 = WavePort(port_b[0], port_b[1], f0=f0, mode_kind="auto",
                      pec=[strip], power=power)
        # Lateral x-walls touch the ground strips; only the air top stays open.
        ABC(air.faces.max(axis="z"))
        line.pec = strip
        line.ports = [p0, p1]

    return line
