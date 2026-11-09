import heapq

import numpy as np
import pytest
import torch

from pathfinder_ml.gpu_plan import components, distance_field, extract_path, path_cost
from pathfinder_ml.grid import (
    OFFSETS, STEP_LEN, Grid, inflate, load_pfmap, maze_world, random_world, save_pfmap)


def dijkstra_field(occ, goal, res):
    """Reference single-goal Dijkstra on the 26-connected grid."""
    nz, ny, nx = occ.shape
    dist = np.full(occ.shape, np.inf)
    gx, gy, gz = goal
    dist[gz, gy, gx] = 0.0
    pq = [(0.0, gx, gy, gz)]
    while pq:
        d, x, y, z = heapq.heappop(pq)
        if d > dist[z, y, x]:
            continue
        for (dx, dy, dz), c in zip(OFFSETS, STEP_LEN * res):
            X, Y, Z = x + dx, y + dy, z + dz
            if 0 <= X < nx and 0 <= Y < ny and 0 <= Z < nz and not occ[Z, Y, X]:
                nd = d + c
                if nd < dist[Z, Y, X] - 1e-12:
                    dist[Z, Y, X] = nd
                    heapq.heappush(pq, (nd, X, Y, Z))
    return dist


def test_pfmap_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    occ = (rng.random((5, 6, 7)) < 0.3).astype(np.uint8)
    g = Grid(occ, (1.0, -2.0, 0.5), 0.2, np.array([[0, 1, 2, 3, 4, 4]], np.int32))
    save_pfmap(tmp_path / "m.pfmap", g)
    h = load_pfmap(tmp_path / "m.pfmap")
    assert h.dims == (7, 6, 5)
    assert np.array_equal(h.occ, occ)
    assert h.min_bound == (1.0, -2.0, 0.5) and h.resolution == 0.2
    assert h.queries.tolist() == [[0, 1, 2, 3, 4, 4]]
    assert h.name == "m"


def test_inflate_is_euclidean_ball():
    raw = np.zeros((9, 9, 9), np.uint8)
    raw[4, 4, 4] = 1
    out = inflate(raw, radius_m=0.2, resolution=0.1)  # r = 2 voxels
    z, y, x = np.nonzero(out)
    r2 = (x - 4) ** 2 + (y - 4) ** 2 + (z - 4) ** 2
    assert r2.max() == 4
    assert len(r2) == sum(
        1 for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)
        if a * a + b * b + c * c <= 4)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_distance_field_matches_dijkstra(seed):
    rng = np.random.default_rng(seed)
    occ = (rng.random((10, 12, 14)) < 0.25).astype(np.uint8)
    goal = (3, 4, 5)
    occ[goal[2], goal[1], goal[0]] = 0
    ref = dijkstra_field(occ, goal, 0.1)
    got = distance_field(torch.from_numpy(occ), torch.tensor([goal]), 0.1, check_every=4)[0]
    got = got.double().numpy()
    assert np.array_equal(np.isinf(ref), np.isinf(got))
    fin = np.isfinite(ref)
    assert np.allclose(ref[fin], got[fin], atol=1e-4)


def test_batched_field_and_path_extraction():
    rng = np.random.default_rng(3)
    occ = (rng.random((2, 8, 10, 12)) < 0.2).astype(np.uint8)
    goals = torch.tensor([[1, 1, 1], [10, 8, 6]])
    for b, g in enumerate(goals.tolist()):
        occ[b, g[2], g[1], g[0]] = 0
    fields = distance_field(torch.from_numpy(occ), goals, 0.1).numpy()
    for b, g in enumerate(goals.tolist()):
        ref = dijkstra_field(occ[b], g, 0.1)
        zz, yy, xx = np.nonzero(np.isfinite(ref) & (ref > 0.5))
        s = (xx[0], yy[0], zz[0])
        p = extract_path(fields[b], s, 0.1)
        assert p[0] == tuple(int(v) for v in s) and p[-1] == tuple(g)
        assert all(occ[b, z, y, x] == 0 for x, y, z in p)
        assert path_cost(p, 0.1) == pytest.approx(ref[s[2], s[1], s[0]], abs=1e-4)


def test_components_separates_regions():
    occ = np.zeros((4, 4, 9), np.uint8)
    occ[:, :, 4] = 1                                  # wall splits x<4 from x>4
    lab = components(torch.from_numpy(occ))[0].numpy()
    assert lab[0, 0, 0] != lab[0, 0, 8]
    assert len(np.unique(lab[lab > 0])) == 2
    assert (lab[:, :, 4] == 0).all()


def test_generators_are_deterministic_and_mostly_free():
    a = random_world(np.random.default_rng(7), size_m=(3.2, 3.2, 2.4))
    b = random_world(np.random.default_rng(7), size_m=(3.2, 3.2, 2.4))
    assert np.array_equal(a.occ, b.occ)
    assert a.dims == (32, 32, 24)
    assert 0.2 < a.occ.mean() < 0.95
    m = maze_world(np.random.default_rng(1), cells=(3, 3, 2))
    assert 0.2 < m.occ.mean() < 0.95
