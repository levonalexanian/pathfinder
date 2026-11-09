"""Actor-critic network for the voxel navigation policy."""

from __future__ import annotations

import torch
import torch.nn as nn

from pathfinder_ml.env import WIN


def _encoder(in_ch=2):
    # 15^3 -> 15^3 -> 8^3 -> 4^3. Narrow at full resolution: rollouts and every
    # PPO epoch run it on each sample, so the first layer dominates the cost.
    return nn.Sequential(
        nn.Conv3d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4 * 4, 256), nn.ReLU(inplace=True),
    )


class ActorCritic(nn.Module):
    def __init__(self, n_actions: int = 26, vec_dim: int = 7):
        super().__init__()
        assert WIN == 15
        self.local = _encoder()
        self.coarse = _encoder()
        self.vec = nn.Sequential(nn.Linear(vec_dim, 64), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(nn.Linear(256 + 256 + 64, 256), nn.ReLU(inplace=True))
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, local, coarse, vec, mask):
        h = torch.cat([self.local(local.float()), self.coarse(coarse.float()), self.vec(vec)], 1)
        h = self.trunk(h)
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(1)
