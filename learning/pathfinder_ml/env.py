"""Batched, GPU-resident RL environment for 26-connected voxel navigation.

Each of ``n_envs`` agents lives in one map of a ``MapBank`` and walks towards a
goal one voxel per step (26 actions; moves into blocked voxels are masked).

Observation (all egocentric, so the policy is map-size agnostic):
  * ``local``  (2, 15, 15, 15): occupancy around the agent at full resolution,
    plus a one-hot goal channel when the goal is inside the window.
  * ``coarse`` (2, 15, 15, 15): the same at 4x coarser resolution (max-pooled),
    covering ~6 m, plus the goal channel.
  * ``vec``    (7,): goal offset in metres, its unit direction, log distance.

Reward uses the exact cost-to-goal field D (from the GPU wavefront, used only
during training). Two variants, both 0 for an optimal step (+ goal bonus):
  * ``excess``:     r = alpha * (D(s) - D(s') - step_len) / res
    The return is -alpha * excess path length in voxels: exactly the objective.
  * ``efficiency``: r = alpha * ((D(s) - D(s')) / step_len - 1)
    Penalises the fraction of each step that is not progress. Under a random
    walk (expected progress 0) every move then costs the same, whereas
    ``excess`` makes axis moves look cheaper than diagonals (-1 vs -sqrt2/3)
    and the policy locks onto them early.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pathfinder_ml.gpu_plan import distance_field
from pathfinder_ml.grid import OFFSETS, STEP_LEN, Grid

WIN = 15
HALF = WIN // 2
COARSE = 4


def _window_offsets(device):
    r = torch.arange(-HALF, HALF + 1, device=device)
    dz, dy, dx = torch.meshgrid(r, r, r, indexing="ij")
    return dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)


class PaddedMaps:
    """Maps padded with blocked voxels (and a coarse max-pooled copy), so any
    15^3 window around an in-map voxel can be gathered without bounds checks."""

    def __init__(self, occs: list[np.ndarray], device):
        self.device = device
        zs, ys, xs = zip(*(o.shape for o in occs))
        Z, Y, X = (int(np.ceil(max(v) / COARSE) * COARSE) for v in (zs, ys, xs))
        self.shape = (Z, Y, X)
        full = torch.ones((len(occs), Z, Y, X), dtype=torch.uint8)
        for i, o in enumerate(occs):
            full[i, :o.shape[0], :o.shape[1], :o.shape[2]] = torch.from_numpy(o)
        self.dims = torch.tensor([[o.shape[2], o.shape[1], o.shape[0]] for o in occs],
                                 device=device)  # (x, y, z) sizes
        self.occ = full.to(device)                                   # (M, Z, Y, X)
        coarse = F.max_pool3d(self.occ.float().unsqueeze(1), COARSE).squeeze(1)
        p = HALF
        self.pad = F.pad(self.occ, (p, p, p, p, p, p), value=1).contiguous()
        self.cpad = F.pad(coarse.to(torch.uint8), (p, p, p, p, p, p), value=1).contiguous()
        self.ox, self.oy, self.oz = _window_offsets(device)

    def _gather(self, vol, m, x, y, z):
        M, Zp, Yp, Xp = vol.shape
        base = m * (Zp * Yp * Xp)
        idx = (base[:, None]
               + (z[:, None] + HALF + self.oz) * (Yp * Xp)
               + (y[:, None] + HALF + self.oy) * Xp
               + (x[:, None] + HALF + self.ox))
        return vol.reshape(-1)[idx].reshape(-1, WIN, WIN, WIN)

    def observe(self, m, pos, goal, resolution):
        """Observation dict for agents in maps ``m`` at ``pos`` heading to ``goal``."""
        x, y, z = pos.unbind(1)
        local = self._gather(self.pad, m, x, y, z)
        cpos = pos // COARSE
        coarse = self._gather(self.cpad, m, cpos[:, 0], cpos[:, 1], cpos[:, 2])
        return {
            "local": torch.stack([local, _goal_onehot(goal - pos)], 1),
            "coarse": torch.stack([coarse, _goal_onehot(goal // COARSE - cpos)], 1),
            "vec": _goal_vec(goal - pos, resolution),
        }

    def blocked(self, m, xyz):
        """True where ``xyz`` (N, 3) is outside its map or occupied."""
        x, y, z = xyz.unbind(1)
        d = self.dims[m]
        inside = (x >= 0) & (y >= 0) & (z >= 0) & (x < d[:, 0]) & (y < d[:, 1]) & (z < d[:, 2])
        Z, Y, X = self.shape
        flat = ((m * Z + z.clamp(0, Z - 1)) * Y + y.clamp(0, Y - 1)) * X + x.clamp(0, X - 1)
        return ~inside | (self.occ.reshape(-1)[flat] != 0)

    def action_mask(self, m, pos, offsets):
        """(N, 26) bool, True where the move is allowed."""
        nb = pos[:, None, :] + offsets[None]
        n = pos.shape[0]
        mm = m[:, None].expand(n, 26).reshape(-1)
        return ~self.blocked(mm, nb.reshape(-1, 3)).reshape(n, 26)


def _goal_onehot(rel):
    n = rel.shape[0]
    out = torch.zeros((n, WIN, WIN, WIN), dtype=torch.uint8, device=rel.device)
    inside = (rel.abs() <= HALF).all(1)
    if inside.any():
        r = rel[inside] + HALF
        out[inside.nonzero(as_tuple=True)[0], r[:, 2], r[:, 1], r[:, 0]] = 1
    return out


def _goal_vec(rel, resolution):
    d = rel.float() * resolution
    n = d.norm(dim=1, keepdim=True)
    return torch.cat([d / 5.0, d / n.clamp_min(1e-6), torch.log1p(n)], 1)


@dataclass
class EnvConfig:
    n_envs: int = 512
    alpha: float = 0.1          # reward scale (see module docstring)
    reward: str = "efficiency"  # "efficiency" or "excess"
    goal_bonus: float = 1.0
    max_steps_factor: float = 3.0  # episode limit = factor * optimal hops + 32
    resolution: float = 0.1


class NavEnv:
    """Vectorised training environment over a bank of (map, goal, cost field)."""

    def __init__(self, grids: list[Grid], goals_per_map: int, cfg: EnvConfig, device,
                 rng: np.random.Generator):
        self.cfg, self.device, self.rng = cfg, device, rng
        self.offsets = torch.as_tensor(OFFSETS, device=device)
        self.step_len = torch.as_tensor(STEP_LEN, device=device, dtype=torch.float32)
        self.load_bank(grids, goals_per_map)
        n = cfg.n_envs
        self.m = torch.zeros(n, dtype=torch.long, device=device)
        self.slot = torch.zeros(n, dtype=torch.long, device=device)
        self.pos = torch.zeros((n, 3), dtype=torch.long, device=device)
        self.goal = torch.zeros((n, 3), dtype=torch.long, device=device)
        self.t = torch.zeros(n, dtype=torch.long, device=device)
        self.limit = torch.zeros(n, dtype=torch.long, device=device)
        self.max_start_dist = 1.0   # curriculum, metres
        self.reset(torch.ones(n, dtype=torch.bool, device=device))

    # -- bank ---------------------------------------------------------------
    def load_bank(self, grids: list[Grid], goals_per_map: int):
        """Pick goals and precompute their exact cost-to-goal fields on the GPU."""
        self.maps = PaddedMaps([g.occ for g in grids], self.device)
        G = goals_per_map
        goals = []
        for g in grids:
            zz, yy, xx = np.nonzero(g.occ == 0)
            pick = self.rng.integers(len(xx), size=G)
            goals.append(np.stack([xx[pick], yy[pick], zz[pick]], 1))
        self.bank_goals = torch.as_tensor(np.concatenate(goals), device=self.device)
        self.bank_map = torch.arange(len(grids), device=self.device).repeat_interleave(G)
        fields = []
        for i in range(0, len(self.bank_goals), 32):
            occ = self.maps.occ[self.bank_map[i:i + 32]]
            fields.append(distance_field(occ, self.bank_goals[i:i + 32],
                                         self.cfg.resolution).to(torch.float16))
        self.fields = torch.cat(fields)  # (M*G, Z, Y, X), inf = unreachable
        # Largest finite distance per goal, for the curriculum's upper end.
        f = self.fields.float().reshape(len(self.fields), -1)
        self.field_max = torch.where(torch.isfinite(f), f, torch.zeros_like(f)).amax(1)

    def D(self, slot, xyz):
        Z, Y, X = self.maps.shape
        x, y, z = xyz.unbind(1)
        flat = ((slot * Z + z) * Y + y) * X + x
        return self.fields.reshape(-1)[flat].float()

    # -- episode control ----------------------------------------------------
    def reset(self, mask):
        idx = mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return
        k = len(idx)
        slot = torch.randint(len(self.bank_goals), (k,), device=self.device)
        m = self.bank_map[slot]
        # Sample candidate starts; keep the first one that is reachable and
        # within the curriculum distance (fall back to any reachable one).
        C = 96
        Z, Y, X = self.maps.shape
        dims = self.maps.dims[m]
        cand = (torch.rand((k, C, 3), device=self.device) * dims[:, None, :]).long()
        flat_slot = slot[:, None].expand(k, C).reshape(-1)
        d = self.D(flat_slot, cand.reshape(-1, 3)).reshape(k, C)
        ok = torch.isfinite(d) & (d > 0)
        near = ok & (d <= self.max_start_dist)
        score = near.float() * 2 + ok.float() + torch.rand_like(d) * 0.5
        best = score.argmax(1)
        start = cand[torch.arange(k, device=self.device), best]
        good = ok[torch.arange(k, device=self.device), best]
        # Rare: no reachable candidate. Start at the goal's neighbour search
        # next time by resampling; mark them to retry on the next step.
        start = torch.where(good[:, None], start, self.bank_goals[slot])
        self.m[idx], self.slot[idx] = m, slot
        self.pos[idx], self.goal[idx] = start, self.bank_goals[slot]
        self.t[idx] = 0
        hops = (self.D(slot, start) / self.cfg.resolution).nan_to_num(0, 0, 0)
        self.limit[idx] = (self.cfg.max_steps_factor * hops + 32).long()

    def observe(self):
        obs = self.maps.observe(self.m, self.pos, self.goal, self.cfg.resolution)
        obs["mask"] = self.maps.action_mask(self.m, self.pos, self.offsets)
        return obs

    def step(self, action):
        cfg = self.cfg
        mask = self.maps.action_mask(self.m, self.pos, self.offsets)
        legal = mask.gather(1, action[:, None]).squeeze(1)
        new = torch.where(legal[:, None], self.pos + self.offsets[action], self.pos)
        d0 = self.D(self.slot, self.pos)
        d1 = self.D(self.slot, new)
        step = self.step_len[action] * cfg.resolution
        if cfg.reward == "excess":
            r = cfg.alpha * (d0 - d1 - step) / cfg.resolution
        else:
            r = cfg.alpha * ((d0 - d1) / step - 1.0)
        r = torch.where(legal, r, torch.full_like(r, -cfg.alpha * 2))
        self.pos = new
        self.t += 1
        reached = (self.pos == self.goal).all(1)
        r = r + reached.float() * cfg.goal_bonus
        timeout = self.t >= self.limit
        done = reached | timeout
        info = {"reached": reached, "timeout": timeout & ~reached}
        self.reset(done)
        return r, done, info
