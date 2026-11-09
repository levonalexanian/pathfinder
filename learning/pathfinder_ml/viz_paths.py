"""Record a Rerun comparison of optimal and RL paths on a benchmark map.

    python -m pathfinder_ml.viz_paths bench/scenarios/demo_world.pfmap \
        --policy learning/models/ppo_planner.pt --out bench/results/demo_world_paths.rrd

Open the .rrd in any Rerun viewer (``rerun file.rrd``, or drag it into the web
viewer). Obstacles are drawn as their surface voxels, coloured by height; each
query gets the optimal path (green) and the RL policy's path (orange), with
start and goal markers. Toggle ``queries/<i>`` to inspect one at a time.
"""

from __future__ import annotations

import argparse

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch

from pathfinder_ml.gpu_plan import distance_field, extract_path, path_cost, pick_device
from pathfinder_ml.grid import load_pfmap
from pathfinder_ml.rl_plan import RLPlanner, load_policy

OPT_COLOR = [60, 200, 90]
RL_COLOR = [255, 140, 30]


def surface_voxels(occ):
    """Blocked voxels with at least one free face neighbour (what you can see)."""
    b = occ.astype(bool)
    pad = np.pad(b, 1, constant_values=True)
    inner = (pad[:-2, 1:-1, 1:-1] & pad[2:, 1:-1, 1:-1] & pad[1:-1, :-2, 1:-1]
             & pad[1:-1, 2:, 1:-1] & pad[1:-1, 1:-1, :-2] & pad[1:-1, 1:-1, 2:])
    z, y, x = np.nonzero(b & ~inner)
    return np.stack([x, y, z], 1)


def to_world(grid, vox):
    v = np.asarray(vox, dtype=np.float64)
    return np.asarray(grid.min_bound) + (v + 0.5) * grid.resolution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", default="bench/results/paths.rrd")
    ap.add_argument("--queries", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = pick_device(a.device)
    grid = load_pfmap(a.map)
    q = grid.queries[: a.queries]
    starts, goals = q[:, :3], q[:, 3:]

    fields = distance_field(torch.from_numpy(grid.occ).to(dev).expand(len(q), *grid.occ.shape),
                            torch.as_tensor(goals), grid.resolution).cpu().numpy()
    optimal = [extract_path(fields[i], starts[i], grid.resolution) for i in range(len(q))]
    rl_paths, _ = RLPlanner(load_policy(a.policy, dev), grid, dev).plan(starts, goals)

    rr.init("pathfinder_paths")
    rr.save(a.out, default_blueprint=rrb.Blueprint(rrb.Spatial3DView(origin="world")))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    surf = surface_voxels(grid.occ)
    h = surf[:, 2] / max(1, grid.occ.shape[0] - 1)
    colors = np.stack([90 + 120 * h, 110 + 60 * h, 160 - 60 * h], 1).astype(np.uint8)
    rr.log("world/obstacles", rr.Points3D(to_world(grid, surf), radii=grid.resolution * 0.45,
                                          colors=colors), static=True)
    for i in range(len(q)):
        base = f"world/queries/{i}"
        rr.log(f"{base}/endpoints", rr.Points3D(to_world(grid, [starts[i], goals[i]]),
                                                radii=0.08, colors=[[80, 160, 255], [255, 60, 60]],
                                                labels=["start", "goal"]), static=True)
        opt = optimal[i]
        rr.log(f"{base}/optimal", rr.LineStrips3D([to_world(grid, opt)], radii=0.025,
                                                  colors=[OPT_COLOR]), static=True)
        rl = rl_paths[i]
        if rl is None:
            rr.log(f"{base}/rl", rr.TextDocument("RL policy failed on this query"), static=True)
            print(f"query {i}: optimal {path_cost(opt, grid.resolution):.2f} m, RL failed")
            continue
        rr.log(f"{base}/rl", rr.LineStrips3D([to_world(grid, rl)], radii=0.025,
                                             colors=[RL_COLOR]), static=True)
        print(f"query {i}: optimal {path_cost(opt, grid.resolution):.2f} m, "
              f"RL {path_cost(rl, grid.resolution):.2f} m")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
