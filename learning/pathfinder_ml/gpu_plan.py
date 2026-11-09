"""GPU pathfinding in PyTorch.

``distance_field`` solves single-goal shortest paths on the 26-connected grid
for a whole batch of (map, goal) pairs at once with synchronous Bellman-Ford
("wavefront") relaxation: every iteration relaxes every cell against all 26
neighbours in parallel. It converges to the exact optimum, i.e. the same cost
A* and Dijkstra find, and one field answers every start for that goal.

``components`` labels connected free space with max-pool propagation; it is
used to pick solvable start/goal pairs.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from pathfinder_ml.grid import OFFSETS, STEP_LEN

INF = float("inf")


def distance_field(occ: torch.Tensor, goals: torch.Tensor, resolution: float = 0.1,
                   max_iters: int | None = None, check_every: int = 16,
                   dtype=torch.float32) -> torch.Tensor:
    """Cost-to-goal (metres) for every cell; ``inf`` where blocked/unreachable.

    occ:   (B, Z, Y, X) bool/uint8, nonzero = blocked.
    goals: (B, 3) long, voxel (x, y, z) per batch item.
    """
    if occ.dim() == 3:
        occ = occ.unsqueeze(0)
    B, Z, Y, X = occ.shape
    dev = occ.device
    free = F.pad(occ == 0, (1, 1, 1, 1, 1, 1), value=False)
    d = torch.full((B, Z + 2, Y + 2, X + 2), INF, device=dev, dtype=dtype)
    b = torch.arange(B, device=dev)
    goals = goals.to(dev).long()
    d[b, goals[:, 2] + 1, goals[:, 1] + 1, goals[:, 0] + 1] = 0.0
    d = torch.where(free, d, torch.full_like(d, INF))
    inner = (slice(None), slice(1, Z + 1), slice(1, Y + 1), slice(1, X + 1))
    free_in = free[inner]
    costs = [float(c) * resolution for c in STEP_LEN]
    shifts = [(int(dx), int(dy), int(dz)) for dx, dy, dz in OFFSETS]
    limit = max_iters if max_iters is not None else 3 * (X + Y + Z) + Z * Y * X // 8
    it = 0
    while it < limit:
        for _ in range(check_every):
            cur = d[inner]
            best = cur
            for (dx, dy, dz), c in zip(shifts, costs):
                nb = d[:, 1 + dz:Z + 1 + dz, 1 + dy:Y + 1 + dy, 1 + dx:X + 1 + dx]
                best = torch.minimum(best, nb + c)
            best = torch.where(free_in, best, cur)
            changed = not torch.equal(best, cur) if _ == check_every - 1 else True
            d[inner] = best
            it += 1
        if not changed:
            break
    return d[inner]


def extract_path(dist: np.ndarray, start, resolution: float = 0.1, max_steps=None):
    """Greedy descent of a distance field from ``start`` (x, y, z) to the goal.

    Returns a list of (x, y, z) voxels, or None if ``start`` is unreachable.
    """
    nz, ny, nx = dist.shape
    x, y, z = (int(v) for v in start)
    if not np.isfinite(dist[z, y, x]):
        return None
    path = [(x, y, z)]
    limit = max_steps or nx * ny * nz
    steps = STEP_LEN * resolution
    while dist[z, y, x] > 0 and len(path) < limit:
        # Step to the neighbour minimising cost-to-go + step, among neighbours
        # strictly closer to the goal (guarantees termination).
        here, best, bxyz = dist[z, y, x], np.inf, None
        for (dx, dy, dz), c in zip(OFFSETS, steps):
            X, Y, Z = x + dx, y + dy, z + dz
            if 0 <= X < nx and 0 <= Y < ny and 0 <= Z < nz and dist[Z, Y, X] < here:
                v = dist[Z, Y, X] + c
                if v < best:
                    best, bxyz = v, (int(X), int(Y), int(Z))
        if bxyz is None:
            break
        x, y, z = bxyz
        path.append(bxyz)
    return path if dist[z, y, x] == 0 else None


def path_cost(path, resolution: float = 0.1) -> float:
    p = np.asarray(path, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * resolution)


def components(occ: torch.Tensor, max_iters: int | None = None) -> torch.Tensor:
    """Connected-component labels (26-connectivity) of free space; 0 = blocked."""
    if occ.dim() == 3:
        occ = occ.unsqueeze(0)
    free = occ == 0
    n = free[0].numel()
    lab = torch.arange(1, n + 1, device=occ.device, dtype=torch.float32)
    lab = lab.reshape(free.shape[1:]).expand_as(free).clone() * free
    limit = max_iters or n
    for i in range(limit):
        nxt = F.max_pool3d(lab.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) * free
        if i % 8 == 7 and torch.equal(nxt, lab):
            break
        lab = nxt
    return lab.long()


def pick_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def heuristic_iters(dims) -> int:
    return int(math.ceil(sum(dims) * 1.5))
