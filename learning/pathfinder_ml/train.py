"""Train the RL voxel planner with PPO.

    python -m pathfinder_ml.train --out learning/runs/ppo [--iters 3000] [--resume]

Everything (maps, cost fields, environments, rollouts, updates) lives on the
GPU. Training worlds are procedurally generated from seeds disjoint from the
benchmark's, and the map bank is partially regenerated every
``--bank-refresh`` iterations so the policy cannot memorise maps. A distance
curriculum widens start/goal separation as the success rate climbs.

The discount is short (gamma = lambda = 0.9) on purpose: the shaped reward
already encodes long-range progress, and a long horizon mostly adds variance.
With gamma = 0.995 the policy first collapses onto the six axis moves (cheaper
than diagonals under a random walk) and stops improving.

Metrics go to ``<out>/metrics.csv`` and a Rerun recording ``<out>/train.rrd``;
checkpoints to ``<out>/last.pt`` (resumable) and ``<out>/policy.pt`` (weights).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from pathfinder_ml.env import EnvConfig, NavEnv
from pathfinder_ml.gpu_plan import pick_device
from pathfinder_ml.grid import maze_world, random_world
from pathfinder_ml.model import ActorCritic

TRAIN_SEED_BASE = 1_000_000


def make_training_maps(rng: np.random.Generator, n: int):
    grids = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.6:
            size = (rng.uniform(4.0, 6.4), rng.uniform(4.0, 6.4), rng.uniform(3.2, 4.8))
            grids.append(random_world(rng, size_m=size, density=rng.uniform(0.5, 1.6)))
        else:
            cells = (int(rng.integers(3, 6)), int(rng.integers(3, 6)), int(rng.integers(1, 3)))
            grids.append(maze_world(rng, cells=cells))
    return grids


def rollout(env, net, T, amp):
    N = env.cfg.n_envs
    dev = env.device
    buf = {
        "local": torch.empty((T, N, 2, 15, 15, 15), dtype=torch.uint8, device=dev),
        "coarse": torch.empty((T, N, 2, 15, 15, 15), dtype=torch.uint8, device=dev),
        "vec": torch.empty((T, N, 7), device=dev),
        "mask": torch.empty((T, N, 26), dtype=torch.bool, device=dev),
        "act": torch.empty((T, N), dtype=torch.long, device=dev),
        "logp": torch.empty((T, N), device=dev),
        "val": torch.empty((T + 1, N), device=dev),
        "rew": torch.empty((T, N), device=dev),
        "done": torch.empty((T, N), device=dev),
    }
    stats = {"episodes": 0, "reached": 0}
    with torch.no_grad():
        for t in range(T):
            obs = env.observe()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                logits, v = net(obs["local"], obs["coarse"], obs["vec"], obs["mask"])
            dist = torch.distributions.Categorical(logits=logits.float())
            a = dist.sample()
            for k in ("local", "coarse", "vec", "mask"):
                buf[k][t] = obs[k]
            buf["act"][t], buf["logp"][t], buf["val"][t] = a, dist.log_prob(a), v.float()
            r, done, info = env.step(a)
            buf["rew"][t], buf["done"][t] = r, done.float()
            stats["episodes"] += int(done.sum())
            stats["reached"] += int(info["reached"].sum())
        obs = env.observe()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            buf["val"][T] = net(obs["local"], obs["coarse"], obs["vec"], obs["mask"])[1].float()
    return buf, stats


def gae(buf, gamma, lam):
    T = buf["rew"].shape[0]
    adv = torch.zeros_like(buf["rew"])
    last = torch.zeros_like(buf["rew"][0])
    for t in reversed(range(T)):
        nonterm = 1.0 - buf["done"][t]
        delta = buf["rew"][t] + gamma * buf["val"][t + 1] * nonterm - buf["val"][t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + buf["val"][:T]


def ppo_update(net, opt, buf, adv, ret, a):
    T, N = buf["act"].shape
    flat = {k: buf[k].reshape(T * N, *buf[k].shape[2:])
            for k in ("local", "coarse", "vec", "mask", "act", "logp")}
    adv = adv.reshape(-1)
    ret = ret.reshape(-1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    total = T * N
    mb = min(a.minibatch, total)
    out = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0}
    n = 0
    for _ in range(a.epochs):
        perm = torch.randperm(total, device=adv.device)
        for i in range(0, total - mb + 1, mb):
            idx = perm[i:i + mb]
            with torch.autocast(adv.device.type, dtype=torch.bfloat16, enabled=a.amp):
                logits, v = net(flat["local"][idx], flat["coarse"][idx],
                                flat["vec"][idx], flat["mask"][idx])
            dist = torch.distributions.Categorical(logits=logits.float())
            logp = dist.log_prob(flat["act"][idx])
            ratio = (logp - flat["logp"][idx]).exp()
            s1 = ratio * adv[idx]
            s2 = ratio.clamp(1 - a.clip, 1 + a.clip) * adv[idx]
            pi_loss = -torch.min(s1, s2).mean()
            v_loss = F.mse_loss(v.float(), ret[idx])
            ent = dist.entropy().mean()
            loss = pi_loss + a.vf_coef * v_loss - a.ent_coef * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            with torch.no_grad():
                kl = (flat["logp"][idx] - logp).mean()
            for k, val in (("pi", pi_loss), ("v", v_loss), ("ent", ent), ("kl", kl)):
                out[k] += float(val.detach())
            n += 1
    return {k: v / max(1, n) for k, v in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="learning/runs/ppo")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--n-envs", type=int, default=1024)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--maps", type=int, default=96)
    ap.add_argument("--goals-per-map", type=int, default=8)
    ap.add_argument("--bank-refresh", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--reward", choices=["efficiency", "excess"], default="efficiency")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-rerun", action="store_true")
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    a = ap.parse_args(argv)

    dev = pick_device(a.device)
    a.amp = dev.type == "cuda" and not a.no_amp
    torch.backends.cudnn.benchmark = True
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(TRAIN_SEED_BASE + a.seed)

    net = ActorCritic().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    start_iter, max_start = 0, 1.0
    ckpt_path = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_iter, max_start = ck["iter"] + 1, ck["max_start_dist"]
        rng = np.random.default_rng(TRAIN_SEED_BASE + a.seed + start_iter)
        print(f"resumed from iter {ck['iter']}")

    t0 = time.time()
    grids = make_training_maps(rng, a.maps)
    env = NavEnv(grids, a.goals_per_map, EnvConfig(n_envs=a.n_envs, reward=a.reward), dev, rng)
    env.max_start_dist = max_start
    print(f"device={dev} bank: {a.maps} maps x {a.goals_per_map} goals "
          f"built in {time.time() - t0:.1f}s", flush=True)

    rec = None
    if not a.no_rerun:
        import rerun as rr
        rec = rr.RecordingStream("pathfinder_ppo")
        rec.save(os.path.join(a.out, "train.rrd"))

    csv_path = os.path.join(a.out, "metrics.csv")
    new_csv = not (a.resume and os.path.exists(csv_path))
    f = open(csv_path, "w" if new_csv else "a", newline="")
    w = csv.writer(f)
    if new_csv:
        w.writerow(["iter", "env_steps", "sps", "success", "reward", "max_start_dist",
                    "pi_loss", "v_loss", "entropy", "kl"])

    steps_per_iter = a.n_envs * a.rollout
    lr0 = a.lr
    for it in range(start_iter, a.iters):
        tic = time.time()
        # Right after a full reset no episode can have timed out yet, so that
        # rollout's success rate is biased high; keep it out of the curriculum.
        just_reset = it == start_iter
        if it > start_iter and it % a.bank_refresh == 0:
            # Replace half the maps with fresh ones (keeps memory flat).
            keep = grids[: a.maps // 2]
            grids = make_training_maps(rng, a.maps - len(keep)) + keep
            env.load_bank(grids, a.goals_per_map)
            env.reset(torch.ones(a.n_envs, dtype=torch.bool, device=dev))
            just_reset = True
        for g in opt.param_groups:
            g["lr"] = lr0 * max(0.1, 1.0 - it / a.iters)
        net.train()
        t_roll0 = time.time()
        buf, st = rollout(env, net, a.rollout, a.amp)
        t_roll = time.time() - t_roll0
        adv, ret = gae(buf, a.gamma, a.lam)
        losses = ppo_update(net, opt, buf, adv, ret, a)
        t_upd = time.time() - t_roll0 - t_roll
        success = st["reached"] / max(1, st["episodes"])
        # Curriculum: grow the start/goal distance once episodes mostly succeed.
        if not just_reset and st["episodes"] > 50 and success > 0.85:
            env.max_start_dist = min(env.max_start_dist * 1.15, 100.0)
        sps = steps_per_iter / (time.time() - tic)
        row = [it, (it + 1) * steps_per_iter, round(sps), round(success, 4),
               round(float(buf["rew"].mean()), 5), round(env.max_start_dist, 3),
               round(losses["pi"], 5), round(losses["v"], 5), round(losses["ent"], 4),
               round(losses["kl"], 5)]
        w.writerow(row)
        f.flush()
        if rec is not None:
            import rerun as rr
            rec.set_time("iter", sequence=it)
            for name, val in zip(["success", "reward", "max_start_dist", "entropy", "kl"],
                                 [row[3], row[4], row[5], row[8], row[9]]):
                rec.log(f"train/{name}", rr.Scalars([float(val)]))
        if it % 10 == 0:
            print(f"it {it:5d}  sps {sps:7.0f}  success {success:.3f}  "
                  f"start<= {env.max_start_dist:5.2f}m  ent {losses['ent']:.3f}  "
                  f"kl {losses['kl']:.4f}  rollout {t_roll:.1f}s update {t_upd:.1f}s",
                  flush=True)
        if it % 25 == 0 or it == a.iters - 1:
            state = {"model": net.state_dict(), "opt": opt.state_dict(), "iter": it,
                     "max_start_dist": env.max_start_dist, "args": vars(a)}
            torch.save(state, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            torch.save({k: v.half() for k, v in net.state_dict().items()},
                       os.path.join(a.out, "policy.pt"))
        if not math.isfinite(losses["v"]):
            raise RuntimeError("value loss diverged")
    f.close()


if __name__ == "__main__":
    main()
