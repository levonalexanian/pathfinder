import json

import numpy as np
import torch

from pathfinder_ml import bench_py, report
from pathfinder_ml.grid import Grid, save_pfmap
from pathfinder_ml.model import ActorCritic
from pathfinder_ml.scenarios import sample_queries


def test_bench_py_runs_and_validates(tmp_path):
    rng = np.random.default_rng(0)
    occ = (rng.random((10, 16, 16)) < 0.15).astype(np.uint8)
    g = Grid(occ, (0, 0, 0), 0.1, name="tiny")
    g.queries = sample_queries(g, 3, rng, min_dist_frac=0.3, device=torch.device("cpu"))
    rows = bench_py.run_map(g, torch.device("cpu"), ActorCritic(), 1,
                            {"torch_wavefront", "torch_wavefront_batched", "rl_policy_batched"})
    by = {}
    for r in rows:
        by.setdefault(r["algo"], []).append(r)
    assert len(by["torch_wavefront"]) == 3 and all(r["success"] for r in by["torch_wavefront"])
    a = [r["cost"] for r in by["torch_wavefront"]]
    b = [r["cost"] for r in by["torch_wavefront_batched"]]
    assert np.allclose(a, b)
    # An untrained policy may fail, but any path it returns must be valid.
    assert all(r["success"] or r["note"] == "no path" for r in by["rl_policy_batched"])


def test_check_path_rejects_bad_paths():
    occ = np.zeros((3, 3, 3), np.uint8)
    occ[1, 1, 1] = 1
    assert bench_py.check_path(occ, [(0, 0, 0), (1, 0, 0)], (0, 0, 0), (1, 0, 0)) == ""
    assert bench_py.check_path(occ, [(0, 0, 0), (2, 0, 0)], (0, 0, 0), (2, 0, 0)) != ""
    assert bench_py.check_path(occ, [(0, 0, 0), (1, 1, 1)], (0, 0, 0), (1, 1, 1)) == "collision"
    assert bench_py.check_path(occ, None, (0, 0, 0), (1, 1, 1)) == "no path"


def test_report_cost_ratio_and_render(tmp_path):
    rows = []
    for q in range(2):
        rows.append({"map": "m", "cells": 8, "algo": "astar", "device": "cpu", "query": q,
                     "success": True, "note": "", "cost": 2.0, "path_len": 3, "nodes": 5,
                     "time_ms": 1.0})
        rows.append({"map": "m", "cells": 8, "algo": "rl_policy", "device": "cuda", "query": q,
                     "success": q == 0, "note": "", "cost": 3.0, "path_len": 4, "nodes": 5,
                     "time_ms": 4.0})
    p = tmp_path / "r.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    s = report.summarise(report.load([p]))
    assert s[("m", "rl_policy")]["ok"] == 1
    assert s[("m", "rl_policy")]["cost_ratio"] == 1.5
    assert s[("m", "astar")]["cost_ratio"] == 1.0
    md = report.render(s, str(tmp_path / "out"))
    assert "RL policy (PPO)" in md and (tmp_path / "out" / "median_time.png").exists()
