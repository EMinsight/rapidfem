"""Incremental display streaming (Hebel 1): self-contained captures
(geometry / mesh / time-domain) are serialised and emitted the instant
rapidfem.show() runs, while the sim+result pairing is deferred to cell end.

These cover the split between _serialize_streamable (per-item, live) and
_serialize_paired (deferred), and the show()-time capture callback, without
needing a solver run.
"""
from __future__ import annotations

import pytest

import rapidfem as rf
from rapidfem import _show_capture
from rapidfem._show_capture import CapturedItem

# The UI serialisers live under the [ui] extra (flask etc.); skip if absent.
api = pytest.importorskip("rapidfem.ui.api")

MM = 1e-3


def test_deferred_kinds_not_streamable():
    # sim/result/eigenmode need cross-item pairing, so they must not be
    # serialised live (return None → handled at cell end).
    assert api._serialize_streamable(CapturedItem("p", object(), "simulation")) is None
    assert api._serialize_streamable(CapturedItem("r", object(), "result")) is None
    assert api._serialize_streamable(CapturedItem("m", object(), "eigenmode")) is None


def test_geometry_streams_at_show_time():
    # The win: serialising in the on_item callback captures the geometry
    # state AT show() time, so a pre-mesh show is "geometry" and a post-mesh
    # show is "mesh", instead of both seeing the final (meshed) state.
    g = rf.Geometry(maxh=5 * MM)
    g.box(1 * MM, 1 * MM, 1 * MM, material=rf.Air())
    kinds = []

    def cb(item):
        evt = api._serialize_streamable(item)
        kinds.append(evt["kind"] if evt else None)

    _show_capture.start_capture(on_item=cb)
    try:
        rf.show(g)                 # before meshing -> geometry preview
        g.mesh(optimize=False)
        rf.show(g)                 # after meshing -> mesh
    finally:
        captured = _show_capture.stop_capture()

    assert kinds == ["geometry", "mesh"]
    assert [c.kind for c in captured] == ["geometry", "geometry"]


def test_combined_equals_stream_plus_paired():
    # The batch serialiser must equal streamable-per-item + paired, so
    # non-streaming callers keep identical behaviour.
    g = rf.Geometry(maxh=5 * MM)
    g.box(1 * MM, 1 * MM, 1 * MM, material=rf.Air())
    _show_capture.start_capture()
    try:
        rf.show(g)
    finally:
        captured = _show_capture.stop_capture()

    combined = api._serialize_captures_for_protocol(captured)
    manual = [e for e in (api._serialize_streamable(c) for c in captured) if e]
    manual += api._serialize_paired(captured)
    assert [c["kind"] for c in combined] == [m["kind"] for m in manual]


def test_on_item_resets_after_stop():
    # A streaming run must not leave a stale callback that fires on a later
    # non-streaming capture.
    fired = []
    _show_capture.start_capture(on_item=lambda it: fired.append(it.kind))
    _show_capture.stop_capture()

    _show_capture.start_capture()  # no callback this time
    g = rf.Geometry(maxh=5 * MM)
    g.box(1 * MM, 1 * MM, 1 * MM)
    rf.show(g)
    _show_capture.stop_capture()
    assert fired == []  # the first run's callback never fired
