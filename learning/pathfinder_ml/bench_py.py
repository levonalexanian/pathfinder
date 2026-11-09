"""Benchmark the PyTorch GPU planner and the RL planner on the .pfmap scenarios.

    python -m pathfinder_ml.bench_py --policy learning/models/ppo_planner.pt \
        --out bench/results/py.jsonl bench/scenarios/*.pfmap

Writes the same JSON-lines schema as the C++ ``pf_bench`` so ``report.py`` can
merge them. Algorithms:
  * torch_wavefront          one query at a time: cost field + path extraction
  * torch_wavefront_batched  all of a map's queries in one batched solve
                             (time_ms = batch time / queries)
  * rl_policy                trained PPO policy, one query at a time
  * rl_policy_batched        all of a map's queries rolled out together
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from pathfinder_ml.gpu_plan import distance_field, extract_path, path_cost, pick_device
from pathfinder_ml.grid import load_pfmap
from pathfinder_ml.rl_plan import RLPlanner, load_policy


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def check_path(occ, path, s, g):
    if path is None:
        return "no path"
    if tuple(path[0]) != tuple(s) or tuple(path[-1]) != tuple(g):
        return "wrong endpoints"
    p = np.asarray(path)
    if (np.abs(np.diff(p, axis=0)).max(1) != 1).any():
        return "non-adjacent step"
    if occ[p[:, 2], p[:, 1], p[:, 0]].any():
        return "collision"
    return ""


def record(grid, algo, dev, qi, path, s, g, ms, nodes):
    note = check_path(grid.occ, path, s, g)
    ok = note == ""
    return {"map": grid.name, "cells": int(grid.occ.size), "algo": algo,
            "device": dev.type, "query": qi, "success": ok, "note": note,
            "cost": path_cost(path, grid.resolution) if ok else 0.0,
            "path_len": len(path) if ok else 0, "nodes": int(nodes), "time_ms": ms}


def run_map(grid, dev, net, repeats, algos):
    occ_t = torch.from_numpy(grid.occ).to(dev)
    res = grid.resolution
    Q = len(grid.queries)
    starts, goals = grid.queries[:, :3], grid.queries[:, 3:]
    rows = []
    # Warm up kernels / cuDNN so the first query is not penalised.
    distance_field(occ_t, torch.as_tensor(goals[:1]), res, max_iters=4)
    if "torch_wavefront" in algos:
        for qi in range(Q):
            times = []
            for _ in range(repeats):
                _sync(dev)
                t0 = time.perf_counter()
                d = distance_field(occ_t, torch.as_tensor(goals[qi:qi + 1]), res)[0]
                path = extract_path(d.cpu().numpy(), starts[qi], res)
                times.append((time.perf_counter() - t0) * 1e3)
            rows.append(record(grid, "torch_wavefront", dev, qi, path, starts[qi], goals[qi],
                               float(np.median(times)), 0))
    if "torch_wavefront_batched" in algos:
        times = []
        for _ in range(repeats):
            _sync(dev)
            t0 = time.perf_counter()
            d = distance_field(occ_t.expand(Q, *occ_t.shape), torch.as_tensor(goals), res)
            dn = d.cpu().numpy()
            paths = [extract_path(dn[i], starts[i], res) for i in range(Q)]
            times.append((time.perf_counter() - t0) * 1e3)
        per = float(np.median(times)) / Q
        for qi in range(Q):
            rows.append(record(grid, "torch_wavefront_batched", dev, qi, paths[qi],
                               starts[qi], goals[qi], per, 0))
    if net is not None:
        planner = RLPlanner(net, grid, dev)
        planner.plan(starts[:1], goals[:1], max_steps=2)  # warm-up
        if "rl_policy" in algos:
            for qi in range(Q):
                times = []
                for _ in range(repeats):
                    _sync(dev)
                    t0 = time.perf_counter()
                    paths, steps = planner.plan(starts[qi:qi + 1], goals[qi:qi + 1])
                    times.append((time.perf_counter() - t0) * 1e3)
                rows.append(record(grid, "rl_policy", dev, qi, paths[0], starts[qi], goals[qi],
                                   float(np.median(times)), steps))
        if "rl_policy_batched" in algos:
            times = []
            for _ in range(repeats):
                _sync(dev)
                t0 = time.perf_counter()
                paths, steps = planner.plan(starts, goals)
                times.append((time.perf_counter() - t0) * 1e3)
            per = float(np.median(times)) / Q
            for qi in range(Q):
                rows.append(record(grid, "rl_policy_batched", dev, qi, paths[qi], starts[qi],
                                   goals[qi], per, steps))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("maps", nargs="+")
    ap.add_argument("--policy", default=None, help="policy weights (.pt); omit to skip RL")
    ap.add_argument("--out", default="bench/results/py.jsonl")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--algos", default="torch_wavefront,torch_wavefront_batched,"
                                       "rl_policy,rl_policy_batched")
    a = ap.parse_args()
    dev = pick_device(a.device)
    algos = set(a.algos.split(","))
    net = load_policy(a.policy, dev) if a.policy else None
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        for path in a.maps:
            grid = load_pfmap(path)
            rows = run_map(grid, dev, net, a.repeats, algos)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            by = {}
            for r in rows:
                by.setdefault(r["algo"], []).append(r)
            for algo, rs in by.items():
                ok = sum(r["success"] for r in rs)
                print(f"{grid.name:12s} {algo:24s} {ok:3d}/{len(rs):<3d} ok  "
                      f"mean {np.mean([r['time_ms'] for r in rs]):9.2f} ms  [{dev.type}]",
                      flush=True)


if __name__ == "__main__":
    main()
