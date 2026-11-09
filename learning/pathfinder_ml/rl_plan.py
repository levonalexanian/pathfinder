"""Use a trained policy as a path planner.

The policy walks greedily (argmax over legal moves). To avoid dithering it
prefers neighbours it has not visited yet; if every legal neighbour was
visited it may backtrack. Planning fails if the goal is not reached within
``max_steps``. Many queries can be solved at once as a batch.
"""

from __future__ import annotations

import torch

from pathfinder_ml.env import PaddedMaps
from pathfinder_ml.grid import OFFSETS, STEP_LEN, Grid
from pathfinder_ml.model import ActorCritic


def load_policy(path, device) -> ActorCritic:
    net = ActorCritic().to(device)
    sd = torch.load(path, map_location=device)
    sd = sd.get("model", sd)
    net.load_state_dict({k: v.float() for k, v in sd.items()})
    net.eval()
    return net


class RLPlanner:
    def __init__(self, net: ActorCritic, grid: Grid, device):
        self.net, self.grid, self.device = net, grid, device
        self.maps = PaddedMaps([grid.occ], device)
        self.offsets = torch.as_tensor(OFFSETS, device=device)
        self.step_len = torch.as_tensor(STEP_LEN, device=device, dtype=torch.float32)

    @torch.no_grad()
    def plan(self, starts, goals, max_steps=None):
        """starts, goals: (Q, 3) voxel xyz. Returns (paths, steps) where paths[i]
        is a list of xyz tuples ending at the goal, or None on failure."""
        dev = self.device
        pos = torch.as_tensor(starts, device=dev, dtype=torch.long).reshape(-1, 3)
        goal = torch.as_tensor(goals, device=dev, dtype=torch.long).reshape(-1, 3)
        Q = len(pos)
        nx, ny, nz = self.grid.dims
        max_steps = max_steps or 4 * (nx + ny + nz)
        m = torch.zeros(Q, dtype=torch.long, device=dev)
        visited = torch.zeros((Q, nz, ny, nx), dtype=torch.bool, device=dev)
        q = torch.arange(Q, device=dev)
        visited[q, pos[:, 2], pos[:, 1], pos[:, 0]] = True
        hist = [pos.clone()]
        done = (pos == goal).all(1)
        res = self.grid.resolution
        for _ in range(max_steps):
            if bool(done.all()):
                break
            obs = self.maps.observe(m, pos, goal, res)
            legal = self.maps.action_mask(m, pos, self.offsets)
            nb = pos[:, None, :] + self.offsets[None]
            nbc = nb.clamp_min(0)
            nbc = torch.minimum(nbc, torch.tensor([nx - 1, ny - 1, nz - 1], device=dev))
            seen = visited[q[:, None], nbc[..., 2], nbc[..., 1], nbc[..., 0]]
            fresh = legal & ~seen
            # Unvisited neighbours if any, otherwise allow backtracking.
            mask = torch.where(fresh.any(1, keepdim=True), fresh, legal)
            logits, _ = self.net(obs["local"], obs["coarse"], obs["vec"], mask)
            a = logits.argmax(1)
            stuck = ~mask.any(1)
            step = torch.where((done | stuck)[:, None], torch.zeros_like(pos), self.offsets[a])
            pos = pos + step
            visited[q, pos[:, 2], pos[:, 1], pos[:, 0]] = True
            hist.append(pos.clone())
            done = done | (pos == goal).all(1)
        traj = torch.stack(hist, 1).cpu().numpy()  # (Q, steps+1, 3)
        reached = done.cpu().numpy()
        paths = []
        for i in range(Q):
            if not reached[i]:
                paths.append(None)
                continue
            p, last = [], None
            for v in map(tuple, traj[i].tolist()):
                if v != last:
                    p.append(v)
                    last = v
            paths.append(_shortcut_loops(p))
        return paths, len(hist) - 1


def _shortcut_loops(path):
    """Drop cycles: if the walk revisits a voxel, cut the loop out."""
    seen, out = {}, []
    for v in path:
        if v in seen:
            del out[seen[v] + 1:]
            seen = {u: i for i, u in enumerate(out)}
            continue
        seen[v] = len(out)
        out.append(v)
    return out
