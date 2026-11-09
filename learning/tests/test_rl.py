import numpy as np
import pytest
import torch

from pathfinder_ml import train
from pathfinder_ml.env import EnvConfig, NavEnv, PaddedMaps
from pathfinder_ml.gpu_plan import distance_field, extract_path
from pathfinder_ml.grid import OFFSETS, Grid
from pathfinder_ml.model import ActorCritic
from pathfinder_ml.rl_plan import RLPlanner, _shortcut_loops

DEV = torch.device("cpu")


def box_world(seed, shape=(12, 16, 16)):
    rng = np.random.default_rng(seed)
    occ = (rng.random(shape) < 0.15).astype(np.uint8)
    occ[:, :, 7] = 1          # a wall with a doorway
    occ[2:6, 4:8, 7] = 0
    return Grid(occ, (0, 0, 0), 0.1)


def test_window_gather_matches_numpy():
    g = box_world(0)
    maps = PaddedMaps([g.occ], DEV)
    pos = torch.tensor([[3, 4, 5], [0, 0, 0], [15, 15, 11]])
    goal = torch.tensor([[5, 4, 5], [15, 15, 11], [0, 0, 0]])
    obs = maps.observe(torch.zeros(3, dtype=torch.long), pos, goal, 0.1)
    assert obs["local"].shape == (3, 2, 15, 15, 15)
    padded = np.pad(g.occ, 7, constant_values=1)
    for i, (x, y, z) in enumerate(pos.tolist()):
        ref = padded[z:z + 15, y:y + 15, x:x + 15]
        assert np.array_equal(obs["local"][i, 0].numpy(), ref)
    # goal one-hot: query 0 has its goal 2 voxels along +x
    assert obs["local"][0, 1, 7, 7, 9] == 1 and obs["local"][0, 1].sum() == 1
    assert obs["local"][1, 1].sum() == 0  # far goal: outside window


def test_action_mask_blocks_occupied_and_out_of_bounds():
    g = box_world(1)
    maps = PaddedMaps([g.occ], DEV)
    pos = torch.tensor([[0, 0, 0]])
    mask = maps.action_mask(torch.zeros(1, dtype=torch.long), pos, torch.as_tensor(OFFSETS))[0]
    for k, (dx, dy, dz) in enumerate(OFFSETS):
        x, y, z = dx, dy, dz
        free = x >= 0 and y >= 0 and z >= 0 and g.occ[z, y, x] == 0
        assert bool(mask[k]) == free


@pytest.mark.parametrize("reward", ["efficiency", "excess"])
def test_optimal_step_has_zero_reward_and_goal_gives_bonus(reward):
    g = box_world(2)
    env = NavEnv([g], 2, EnvConfig(n_envs=4, reward=reward), DEV, np.random.default_rng(0))
    env.max_start_dist = 100.0
    env.reset(torch.ones(4, dtype=torch.bool))
    field = env.fields[env.slot[0]].float().numpy()
    path = extract_path(field, env.pos[0].tolist(), 0.1)
    nxt = np.array(path[1]) - np.array(path[0])
    a0 = int(np.flatnonzero((OFFSETS == nxt).all(1))[0])
    actions = torch.zeros(4, dtype=torch.long)
    actions[0] = a0
    was_goal = len(path) == 2
    r, done, info = env.step(actions)
    expected = env.cfg.goal_bonus if was_goal else 0.0
    assert float(r[0]) == pytest.approx(expected, abs=1e-2)


def test_shortcut_loops():
    p = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 0, 0), (1, 1, 0)]
    assert _shortcut_loops(p) == [(0, 0, 0), (1, 0, 0), (1, 1, 0)]


def test_rl_planner_paths_are_valid_when_found():
    g = box_world(3)
    net = ActorCritic()
    planner = RLPlanner(net, g, DEV)
    zz, yy, xx = np.nonzero(g.occ == 0)
    s, t = (xx[0], yy[0], zz[0]), (xx[-1], yy[-1], zz[-1])
    paths, _ = planner.plan([s], [t], max_steps=300)
    if paths[0] is not None:
        p = np.array(paths[0])
        assert tuple(p[0]) == s and tuple(p[-1]) == t
        assert (np.abs(np.diff(p, axis=0)).max(1) == 1).all()
        assert not g.occ[p[:, 2], p[:, 1], p[:, 0]].any()


def test_train_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "make_training_maps",
                        lambda rng, n: [box_world(int(rng.integers(1 << 30))) for _ in range(n)])
    train.main(["--out", str(tmp_path), "--iters", "3", "--n-envs", "16", "--rollout", "8",
                "--maps", "2", "--goals-per-map", "2", "--minibatch", "64", "--epochs", "1",
                "--bank-refresh", "2", "--device", "cpu", "--no-rerun"])
    assert (tmp_path / "policy.pt").exists()
    rows = (tmp_path / "metrics.csv").read_text().strip().splitlines()
    assert len(rows) == 4
    train.main(["--out", str(tmp_path), "--iters", "4", "--n-envs", "16", "--rollout", "8",
                "--maps", "2", "--goals-per-map", "2", "--minibatch", "64", "--epochs", "1",
                "--device", "cpu", "--no-rerun", "--resume"])
    assert len((tmp_path / "metrics.csv").read_text().strip().splitlines()) == 5
