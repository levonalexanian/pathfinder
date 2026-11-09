from types import SimpleNamespace as NS

import numpy as np
import rerun as rr

from pathfinder_rerun import markers as mk


def pt(x, y, z):
    return NS(x=x, y=y, z=z)


def color(r, g, b, a):
    return NS(r=r, g=g, b=b, a=a)


def marker(type_, action=mk.ADD, ns="ns", id_=0, points=(), colors=(), scale=(1, 1, 1)):
    return NS(
        type=type_, action=action, ns=ns, id=id_,
        points=list(points), colors=list(colors),
        color=color(1.0, 0.5, 0.0, 1.0),
        scale=NS(x=scale[0], y=scale[1], z=scale[2]),
        pose=NS(position=pt(1, 2, 3), orientation=NS(x=0, y=0, z=0, w=1)),
    )


def test_entity_path_sanitizes_namespace():
    assert mk.entity_path("world/search", "open set/1", 3) == "world/search/open_set_1/3"
    assert mk.entity_path("root", "", 0) == "root/default/0"


def test_colors_use_marker_color_when_no_per_point_colors():
    m = marker(mk.CUBE_LIST, points=[pt(0, 0, 0), pt(1, 1, 1)])
    out = mk._colors(m, 2)
    assert out.dtype == np.uint8
    assert out.tolist() == [[255, 128, 0, 255]]


def test_colors_per_point():
    m = marker(mk.CUBE_LIST, points=[pt(0, 0, 0)], colors=[color(0, 1, 0, 0.5)])
    assert mk._colors(m, 1).tolist() == [[0, 255, 0, 128]]


def test_supported_types_map_to_archetypes():
    pts = [pt(0, 0, 0), pt(1, 0, 0), pt(1, 1, 0), pt(0, 1, 0)]
    assert isinstance(mk.to_archetype(marker(mk.CUBE)), rr.Boxes3D)
    assert isinstance(mk.to_archetype(marker(mk.CUBE_LIST, points=pts)), rr.Boxes3D)
    assert isinstance(mk.to_archetype(marker(mk.LINE_STRIP, points=pts)), rr.LineStrips3D)
    assert isinstance(mk.to_archetype(marker(mk.LINE_LIST, points=pts)), rr.LineStrips3D)
    assert isinstance(mk.to_archetype(marker(mk.POINTS, points=pts)), rr.Points3D)
    assert mk.to_archetype(marker(mk.ARROW)) is None


def test_degenerate_line_strip_clears():
    assert isinstance(mk.to_archetype(marker(mk.LINE_STRIP, points=[pt(0, 0, 0)])), rr.Clear)


def test_marker_array_ops_handles_delete_actions():
    ops = mk.marker_array_ops(
        [
            marker(mk.CUBE, ns="a", id_=1),
            marker(mk.CUBE, action=mk.DELETE, ns="a", id_=2),
            marker(mk.CUBE, action=mk.DELETEALL),
            marker(mk.ARROW, ns="a", id_=3),
        ],
        "world/search",
    )
    paths = [p for p, _ in ops]
    assert paths == ["world/search/a/1", "world/search/a/2", "world/search"]
    assert isinstance(ops[1][1], rr.Clear)
    assert isinstance(ops[2][1], rr.Clear)
