"""Voxel grids, the ``.pfmap`` scenario format, and procedural worlds.

A grid is a dense uint8 occupancy array of shape ``(nz, ny, nx)`` in C order,
so its flat index is ``(z * ny + y) * nx + x``, identical to
``pathfinder_core::InflatedVoxelGrid::linear_index``. Occupancy is already
inflated by the robot radius, exactly like the C++ planners expect.

``.pfmap`` layout (little endian), shared with the C++ benchmark harness:

    char[8]   magic "PFMAP001"
    int32[3]  dims (nx, ny, nz)
    float64[3] min bound (m)
    float64   resolution (m)
    uint8[nx*ny*nz] occupancy
    int32     query count Q
    int32[Q][6] queries (sx, sy, sz, gx, gy, gz) in voxel coordinates
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"PFMAP001"

# 26-connected neighbourhood, same order as kNeighborOffsets26 in voxel_grid.hpp
# (dx outermost, dz innermost).
OFFSETS = np.array(
    [(dx, dy, dz)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if (dx, dy, dz) != (0, 0, 0)],
    dtype=np.int64,
)
STEP_LEN = np.sqrt((OFFSETS ** 2).sum(axis=1))  # in voxels: 1, sqrt2, sqrt3


@dataclass
class Grid:
    occ: np.ndarray                      # uint8 (nz, ny, nx), 1 = blocked
    min_bound: tuple = (0.0, 0.0, 0.0)
    resolution: float = 0.1
    queries: np.ndarray = field(default_factory=lambda: np.zeros((0, 6), np.int32))
    name: str = ""

    @property
    def dims(self):
        nz, ny, nx = self.occ.shape
        return nx, ny, nz

    def free(self, x, y, z) -> bool:
        nx, ny, nz = self.dims
        return 0 <= x < nx and 0 <= y < ny and 0 <= z < nz and self.occ[z, y, x] == 0


def save_pfmap(path, grid: Grid):
    nx, ny, nz = grid.dims
    q = np.asarray(grid.queries, dtype="<i4").reshape(-1, 6)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<3i", nx, ny, nz))
        f.write(struct.pack("<3d", *grid.min_bound))
        f.write(struct.pack("<d", grid.resolution))
        f.write(np.ascontiguousarray(grid.occ, dtype=np.uint8).tobytes())
        f.write(struct.pack("<i", len(q)))
        f.write(q.tobytes())


def load_pfmap(path) -> Grid:
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC:
        raise ValueError(f"{path}: not a pfmap file")
    off = 8
    nx, ny, nz = struct.unpack_from("<3i", data, off)
    off += 12
    mn = struct.unpack_from("<3d", data, off)
    off += 24
    (res,) = struct.unpack_from("<d", data, off)
    off += 8
    n = nx * ny * nz
    occ = np.frombuffer(data, np.uint8, n, off).reshape(nz, ny, nx).copy()
    off += n
    (nq,) = struct.unpack_from("<i", data, off)
    off += 4
    q = np.frombuffer(data, "<i4", nq * 6, off).reshape(nq, 6).astype(np.int32)
    import os
    return Grid(occ, tuple(mn), res, q, os.path.splitext(os.path.basename(str(path)))[0])


# ---------------------------------------------------------------------------
# Inflation
# ---------------------------------------------------------------------------

def inflate(raw: np.ndarray, radius_m: float, resolution: float) -> np.ndarray:
    """Dilate occupancy by a Euclidean ball, matching InflatedVoxelGrid.

    The C++ code inflates by ``ceil(radius / res)`` voxels and keeps cells whose
    integer offset satisfies ``dx^2 + dy^2 + dz^2 <= r^2``.
    """
    r = int(np.ceil(max(0.0, radius_m) / resolution))
    out = raw.astype(bool).copy()
    if r == 0:
        return out.astype(np.uint8)
    src = raw.astype(bool)
    nz, ny, nx = src.shape
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy + dz * dz > r * r or (dx, dy, dz) == (0, 0, 0):
                    continue
                out[max(0, dz):nz + min(0, dz), max(0, dy):ny + min(0, dy),
                    max(0, dx):nx + min(0, dx)] |= \
                    src[max(0, -dz):nz + min(0, -dz), max(0, -dy):ny + min(0, -dy),
                        max(0, -dx):nx + min(0, -dx)]
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Procedural worlds
# ---------------------------------------------------------------------------

def _box(raw, res, x0, y0, z0, x1, y1, z1):
    """Mark the voxels whose centres lie inside the metric box [p0, p1)."""
    nz, ny, nx = raw.shape

    def rng(a, b, n):
        lo = max(0, int(np.ceil(a / res - 0.5)))
        hi = min(n, int(np.ceil(b / res - 0.5)))
        return slice(lo, max(lo, hi))

    raw[rng(z0, z1, nz), rng(y0, y1, ny), rng(x0, x1, nx)] = 1


def random_world(rng: np.random.Generator, size_m=(6.4, 6.4, 4.8), res=0.1,
                 robot_radius=0.25, density=1.0) -> Grid:
    """A cluttered indoor-ish world in the spirit of the demo map.

    Floor, ceiling, pillars, floating slabs (overhangs), and walls with doorways.
    ``density`` scales the obstacle counts.
    """
    sx, sy, sz = size_m
    nx, ny, nz = (int(round(s / res)) for s in size_m)
    raw = np.zeros((nz, ny, nx), np.uint8)
    _box(raw, res, 0, 0, 0, sx, sy, res)                 # floor
    _box(raw, res, 0, 0, sz - res, sx, sy, sz)           # ceiling
    area = sx * sy / 40.0
    for _ in range(rng.poisson(6 * area * density)):     # pillars
        w = rng.uniform(0.2, 0.8)
        d = rng.uniform(0.2, 0.8)
        x, y = rng.uniform(0, sx), rng.uniform(0, sy)
        _box(raw, res, x, y, 0, x + w, y + d, rng.uniform(0.5, sz))
    for _ in range(rng.poisson(4 * area * density)):     # slabs / overhangs
        w, d = rng.uniform(0.6, 2.5, size=2)
        x, y = rng.uniform(-0.5, sx), rng.uniform(-0.5, sy)
        z = rng.uniform(0.6, sz - 0.6)
        _box(raw, res, x, y, z, x + w, y + d, z + rng.uniform(0.1, 0.4))
    for _ in range(rng.poisson(2 * area * density)):     # walls with a doorway
        t = rng.uniform(0.1, 0.25)
        h = rng.uniform(1.0, sz)
        if rng.random() < 0.5:
            y = rng.uniform(0.5, sy - 0.5)
            x0, x1 = sorted(rng.uniform(0, sx, size=2))
            gap = rng.uniform(x0, x1)
            gw = rng.uniform(0.8, 1.5)
            _box(raw, res, x0, y, 0, gap, y + t, h)
            _box(raw, res, gap + gw, y, 0, x1, y + t, h)
        else:
            x = rng.uniform(0.5, sx - 0.5)
            y0, y1 = sorted(rng.uniform(0, sy, size=2))
            gap = rng.uniform(y0, y1)
            gw = rng.uniform(0.8, 1.5)
            _box(raw, res, x, y0, 0, x + t, gap, h)
            _box(raw, res, x, gap + gw, 0, x + t, y1, h)
    return Grid(inflate(raw, robot_radius, res), (0.0, 0.0, 0.0), res)


def maze_world(rng: np.random.Generator, cells=(8, 8, 2), cell_m=1.2, res=0.1,
               robot_radius=0.25) -> Grid:
    """A multi-storey grid maze: hard for greedy and local-view planners."""
    cx, cy, cz = cells
    wall = 0.1
    sx, sy, sz = cx * cell_m, cy * cell_m, cz * cell_m * 1.5
    nx, ny, nz = (int(round(s / res)) for s in (sx, sy, sz))
    raw = np.zeros((nz, ny, nx), np.uint8)
    floor_h = sz / cz
    # Every cell starts walled; carve a spanning tree (randomised DFS) in 3D.
    visited = np.zeros(cells, bool)
    open_e = set()
    stack = [(0, 0, 0)]
    visited[0, 0, 0] = True
    while stack:
        c = stack[-1]
        nbrs = []
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if all(0 <= n[i] < cells[i] for i in range(3)) and not visited[n]:
                nbrs.append(n)
        if not nbrs:
            stack.pop()
            continue
        n = nbrs[rng.integers(len(nbrs))]
        visited[n] = True
        open_e.add(frozenset((c, n)))
        stack.append(n)
    # Extra loops so there are multiple routes.
    for _ in range(cx * cy * cz // 6):
        a = (rng.integers(cx), rng.integers(cy), rng.integers(cz))
        b = (min(cx - 1, a[0] + 1), a[1], a[2])
        if a != b:
            open_e.add(frozenset((a, b)))
    for k in range(cz + 1):                           # floors / ceilings
        z = k * floor_h
        _box(raw, res, 0, 0, max(0, z - wall), sx, sy, min(sz, z + wall))
    for i in range(cx + 1):
        for j in range(cy):
            for k in range(cz):
                if 0 < i < cx and frozenset(((i - 1, j, k), (i, j, k))) in open_e:
                    continue
                _box(raw, res, i * cell_m - wall / 2, j * cell_m, k * floor_h,
                     i * cell_m + wall / 2, (j + 1) * cell_m, (k + 1) * floor_h)
    for j in range(cy + 1):
        for i in range(cx):
            for k in range(cz):
                if 0 < j < cy and frozenset(((i, j - 1, k), (i, j, k))) in open_e:
                    continue
                _box(raw, res, i * cell_m, j * cell_m - wall / 2, k * floor_h,
                     (i + 1) * cell_m, j * cell_m + wall / 2, (k + 1) * floor_h)
    for i in range(cx):                               # holes between storeys
        for j in range(cy):
            for k in range(cz - 1):
                if frozenset(((i, j, k), (i, j, k + 1))) in open_e:
                    z = (k + 1) * floor_h
                    raw[int((z - wall) / res) - 1:int((z + wall) / res) + 2,
                        int((j * cell_m + 0.1) / res):int(((j + 1) * cell_m - 0.1) / res),
                        int((i * cell_m + 0.1) / res):int(((i + 1) * cell_m - 0.1) / res)] = 0
    return Grid(inflate(raw, robot_radius, res), (0.0, 0.0, 0.0), res)
