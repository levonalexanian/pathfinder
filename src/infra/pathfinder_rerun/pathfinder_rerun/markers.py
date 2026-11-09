"""Convert visualization_msgs/Marker messages into Rerun archetypes.

Kept free of rclpy so it can be unit-tested with plain stand-in objects: every
function only reads attributes (``m.type``, ``m.points[i].x`` ...).
"""

import re

import numpy as np
import rerun as rr

# visualization_msgs/Marker type and action constants.
ARROW, CUBE, SPHERE, CYLINDER = 0, 1, 2, 3
LINE_STRIP, LINE_LIST, CUBE_LIST, SPHERE_LIST, POINTS = 4, 5, 6, 7, 8
ADD, DELETE, DELETEALL = 0, 2, 3

_SAFE = re.compile(r"[^A-Za-z0-9_\-]")


def entity_path(root, ns, marker_id):
    """Stable Rerun entity path for one marker: ``<root>/<ns>/<id>``."""
    ns = _SAFE.sub("_", ns) if ns else "default"
    return f"{root}/{ns}/{marker_id}"


def _points(m):
    return np.array([[p.x, p.y, p.z] for p in m.points], dtype=np.float32).reshape(-1, 3)


def _colors(m, n):
    """Per-element colours if the marker carries them, else the marker colour."""
    if len(m.colors) == n and n > 0:
        rgba = np.array([[c.r, c.g, c.b, c.a] for c in m.colors], dtype=np.float32)
    else:
        c = m.color
        rgba = np.array([[c.r, c.g, c.b, c.a]], dtype=np.float32)
    return (np.clip(rgba, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _pose(m):
    p, q = m.pose.position, m.pose.orientation
    return [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]


def to_archetype(m):
    """Return the Rerun archetype for an ADD marker, or None if unsupported."""
    t = m.type
    if t == CUBE:
        center, quat = _pose(m)
        return rr.Boxes3D(
            centers=[center],
            half_sizes=[[m.scale.x / 2, m.scale.y / 2, m.scale.z / 2]],
            quaternions=[rr.Quaternion(xyzw=quat)],
            colors=_colors(m, 1),
            fill_mode="solid",
        )
    if t == CUBE_LIST:
        pts = _points(m)
        half = [m.scale.x / 2, m.scale.y / 2, m.scale.z / 2]
        return rr.Boxes3D(
            centers=pts,
            half_sizes=np.tile(half, (len(pts), 1)),
            colors=_colors(m, len(pts)),
            fill_mode="solid",
        )
    if t == LINE_STRIP:
        pts = _points(m)
        if len(pts) < 2:
            return rr.Clear(recursive=False)
        return rr.LineStrips3D([pts], radii=[m.scale.x / 2], colors=_colors(m, 1)[:1])
    if t == LINE_LIST:
        pts = _points(m)
        segs = pts[: len(pts) // 2 * 2].reshape(-1, 2, 3)
        return rr.LineStrips3D(segs, radii=[m.scale.x / 2], colors=_colors(m, 1)[:1])
    if t in (SPHERE_LIST, POINTS):
        pts = _points(m)
        return rr.Points3D(pts, radii=[m.scale.x / 2], colors=_colors(m, len(pts)))
    if t == SPHERE:
        center, _ = _pose(m)
        return rr.Points3D([center], radii=[m.scale.x / 2], colors=_colors(m, 1))
    return None


def marker_array_ops(markers, root):
    """Translate a MarkerArray into a list of ``(entity_path, archetype)`` logs.

    DELETEALL clears everything under ``root``; DELETE clears one marker.
    Unsupported marker types are skipped.
    """
    ops = []
    for m in markers:
        if m.action == DELETEALL:
            ops.append((root, rr.Clear(recursive=True)))
            continue
        path = entity_path(root, m.ns, m.id)
        if m.action == DELETE:
            ops.append((path, rr.Clear(recursive=False)))
            continue
        arch = to_archetype(m)
        if arch is not None:
            ops.append((path, arch))
    return ops
