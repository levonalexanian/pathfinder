"""Merge benchmark JSON lines into a markdown report and charts.

    python -m pathfinder_ml.report bench/results/*.jsonl --out docs/benchmarks

For every query the reference cost is the cheapest valid path any *exact*
planner found (A*, Dijkstra, D* Lite, CUDA, torch), so "cost ratio" shows how
much longer RRT* and the RL policy's paths are than optimal.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

EXACT = {"astar", "dijkstra", "dstar_lite", "cuda_wavefront", "cuda_frontier",
         "torch_wavefront", "torch_wavefront_batched"}
ORDER = ["astar", "dijkstra", "dstar_lite", "rrt_star", "cuda_frontier", "cuda_wavefront",
         "torch_wavefront", "torch_wavefront_batched", "rl_policy", "rl_policy_batched"]
LABEL = {
    "astar": "A* (CPU)", "dijkstra": "Dijkstra (CPU)", "dstar_lite": "D* Lite (CPU)",
    "rrt_star": "RRT* (CPU)", "cuda_frontier": "CUDA frontier A*-pruned",
    "cuda_wavefront": "CUDA tiled wavefront", "torch_wavefront": "PyTorch wavefront",
    "torch_wavefront_batched": "PyTorch wavefront, batched",
    "rl_policy": "RL policy (PPO)", "rl_policy_batched": "RL policy, batched",
}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def summarise(rows):
    best = {}
    for r in rows:
        if r["success"] and r["algo"] in EXACT:
            k = (r["map"], r["query"])
            best[k] = min(best.get(k, np.inf), r["cost"])
    groups = defaultdict(list)
    for r in rows:
        groups[(r["map"], r["algo"])].append(r)
    out = {}
    for (m, algo), rs in groups.items():
        ok = [r for r in rs if r["success"]]
        ratios = [r["cost"] / best[(m, r["query"])] for r in ok
                  if best.get((m, r["query"]), 0) > 0]
        times = [r["time_ms"] for r in rs]
        out[(m, algo)] = {
            "n": len(rs), "ok": len(ok), "device": rs[0]["device"], "cells": rs[0]["cells"],
            "median_ms": float(np.median(times)), "mean_ms": float(np.mean(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "cost_ratio": float(np.mean(ratios)) if ratios else float("nan"),
            "nodes": float(np.mean([r["nodes"] for r in rs])),
        }
    return out


def overall(summary, maps, algos):
    """Per planner across all maps: successes, geometric-mean speed-up over A*
    (per-map median times), and mean cost ratio."""
    rows = []
    for a in algos:
        ss = [summary[(m, a)] for m in maps if (m, a) in summary]
        ok, n = sum(s["ok"] for s in ss), sum(s["n"] for s in ss)
        sp = [summary[(m, "astar")]["median_ms"] / summary[(m, a)]["median_ms"]
              for m in maps if (m, a) in summary and (m, "astar") in summary
              and summary[(m, a)]["median_ms"] > 0]
        ratios = [s["cost_ratio"] for s in ss if not np.isnan(s["cost_ratio"])]
        rows.append((a, ok, n, float(np.exp(np.mean(np.log(sp)))) if sp else float("nan"),
                     float(np.mean(ratios)) if ratios else float("nan")))
    return rows


def render(summary, out_dir, title_note="", training_png=None):
    os.makedirs(out_dir, exist_ok=True)
    maps = sorted({m for m, _ in summary}, key=lambda m: (m != "demo_world", m))
    algos = [a for a in ORDER if any((m, a) in summary for m in maps)]
    lines = ["# Planner benchmark", ""]
    if title_note:
        lines += [title_note, ""]
    lines += ["Median wall time per query (ms), success rate, and path cost relative to "
              "the optimal 26-connected path (1.000 = optimal). Batched rows report "
              "batch time divided by the number of queries.", "",
              "## Summary (all maps)", "",
              "| planner | success | speed-up vs A* (geo. mean) | mean cost ratio |",
              "|---|---|---:|---:|"]
    for a, ok, n, sp, ratio in overall(summary, maps, algos):
        sp_s = "n/a" if np.isnan(sp) else f"{sp:.2f}x"
        r_s = "n/a" if np.isnan(ratio) else f"{ratio:.3f}"
        lines.append(f"| {LABEL.get(a, a)} | {ok}/{n} | {sp_s} | {r_s} |")
    lines += ["", "![median time per query](median_time.png)", ""]
    if training_png:
        lines += ["![RL training curve](training.png)", ""]
    for m in maps:
        cells = next(summary[(m, a)]["cells"] for a in algos if (m, a) in summary)
        lines += [f"## {m} ({cells:,} voxels)", "",
                  "| planner | device | success | median ms | p95 ms | cost ratio |",
                  "|---|---|---|---:|---:|---:|"]
        for a in algos:
            s = summary.get((m, a))
            if not s:
                continue
            ratio = "n/a" if np.isnan(s["cost_ratio"]) else f"{s['cost_ratio']:.3f}"
            lines.append(f"| {LABEL.get(a, a)} | {s['device']} | {s['ok']}/{s['n']} | "
                         f"{s['median_ms']:.2f} | {s['p95_ms']:.2f} | {ratio} |")
        lines.append("")
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("\n".join(lines))
    _chart(summary, maps, algos, os.path.join(out_dir, "median_time.png"))
    return "\n".join(lines)


def _chart(summary, maps, algos, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(maps)), 5))
    w = 0.8 / max(1, len(algos))
    cmap = plt.get_cmap("tab10")
    for i, a in enumerate(algos):
        xs, ys = [], []
        for j, m in enumerate(maps):
            s = summary.get((m, a))
            if s:
                xs.append(j + (i - len(algos) / 2 + 0.5) * w)
                ys.append(max(s["median_ms"], 1e-3))
        ax.bar(xs, ys, w, label=LABEL.get(a, a), color=cmap(i % 10))
    ax.set_yscale("log")
    ax.set_xticks(range(len(maps)), maps, rotation=20)
    ax.set_ylabel("median time per query (ms, log)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def training_chart(metrics_csv, path):
    """Success rate and curriculum distance over PPO iterations."""
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(metrics_csv) as f:
        rows = list(csv.DictReader(f))
    it = np.array([int(r["iter"]) for r in rows])
    succ = np.array([float(r["success"]) for r in rows])
    dist = np.array([float(r["max_start_dist"]) for r in rows])
    k = 20
    smooth = np.convolve(succ, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(it[k - 1:], smooth, color="tab:green", label=f"episode success ({k}-iter mean)")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("PPO iteration (32k env steps each)")
    ax.set_ylabel("success")
    ax2 = ax.twinx()
    ax2.plot(it, np.minimum(dist, 15), color="tab:gray", alpha=0.6, label="curriculum start distance")
    ax2.set_ylabel("max start distance (m, capped)")
    fig.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="docs/benchmarks")
    ap.add_argument("--note", default="")
    ap.add_argument("--training-metrics", default=None, help="PPO metrics.csv to plot")
    a = ap.parse_args()
    png = None
    if a.training_metrics:
        os.makedirs(a.out, exist_ok=True)
        png = os.path.join(a.out, "training.png")
        training_chart(a.training_metrics, png)
    print(render(summarise(load(a.results)), a.out, a.note, png))


if __name__ == "__main__":
    main()
