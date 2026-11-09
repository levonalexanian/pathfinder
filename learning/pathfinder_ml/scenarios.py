"""Generate the benchmark scenario set (``.pfmap`` files with queries).

    python -m pathfinder_ml.scenarios --out bench/scenarios --demo build/demo_world.pfmap

Every query is solvable (start and goal in the same free component) and the
whole set is seeded, so CPU, CUDA, torch and RL planners see identical inputs.
Evaluation seeds are disjoint from the training seeds used by the RL trainer.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from pathfinder_ml.gpu_plan import components, pick_device
from pathfinder_ml.grid import Grid, load_pfmap, maze_world, random_world, save_pfmap

EVAL_SEED = 20261109


def sample_queries(grid: Grid, n: int, rng: np.random.Generator, min_dist_frac=0.35,
                   device=None) -> np.ndarray:
    """``n`` solvable (start, goal) voxel pairs at least ``min_dist_frac`` of the
    map diagonal apart, both inside the largest free component."""
    device = device or pick_device()
    lab = components(torch.from_numpy(grid.occ).to(device))[0].cpu().numpy()
    ids, counts = np.unique(lab[lab > 0], return_counts=True)
    if len(ids) == 0:
        raise ValueError(f"{grid.name}: no free space")
    zz, yy, xx = np.nonzero(lab == ids[np.argmax(counts)])
    cells = np.stack([xx, yy, zz], axis=1)
    diag = float(np.linalg.norm(grid.dims))
    out = []
    for _ in range(200 * n):
        a, b = cells[rng.integers(len(cells), size=2)]
        if np.linalg.norm(a - b) >= min_dist_frac * diag:
            out.append(np.concatenate([a, b]))
            if len(out) == n:
                break
    if len(out) < n:
        raise ValueError(f"{grid.name}: could only place {len(out)}/{n} queries")
    return np.asarray(out, np.int32)


def build(out_dir: str, demo: str | None, queries: int = 8, quick: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(EVAL_SEED)
    sets: list[Grid] = []
    if demo:
        g = load_pfmap(demo)
        g.name = "demo_world"
        sets.append(g)
    n_small, n_large, n_maze = (1, 0, 1) if quick else (4, 2, 2)
    for i in range(n_small):
        g = random_world(rng, size_m=(6.4, 6.4, 4.8))
        g.name = f"random_s{i}"
        sets.append(g)
    for i in range(n_large):
        g = random_world(rng, size_m=(20.0, 20.0, 6.0))
        g.name = f"random_l{i}"
        sets.append(g)
    for i in range(n_maze):
        g = maze_world(rng, cells=(6, 6, 2))
        g.name = f"maze{i}"
        sets.append(g)
    for g in sets:
        nq = queries * (4 if g.name == "demo_world" else 1)
        g.queries = sample_queries(g, nq, rng)
        path = os.path.join(out_dir, f"{g.name}.pfmap")
        save_pfmap(path, g)
        nx, ny, nz = g.dims
        print(f"{path}: {nx}x{ny}x{nz} ({g.occ.mean() * 100:.0f}% blocked), {nq} queries")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="bench/scenarios")
    ap.add_argument("--demo", default=None, help="demo world .pfmap exported by pf_export_bt")
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="tiny set for smoke tests")
    a = ap.parse_args()
    build(a.out, a.demo, a.queries, a.quick)


if __name__ == "__main__":
    main()
