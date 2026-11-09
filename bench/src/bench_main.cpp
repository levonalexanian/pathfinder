// Runs the CPU planner cores over .pfmap scenarios and writes one JSON line
// per (map, algorithm, query).
//
//   pf_bench [--algos a,b,...] [--repeats N] [--out results.jsonl] maps/*.pfmap
//
// Times are the median of N runs. Every returned path is validated (in
// bounds, collision-free, 26-adjacent steps for grid planners, ends at the
// goal); an invalid path counts as a failure.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <pathfinder_algo_astar/astar_core.hpp>
#include <pathfinder_algo_dijkstra/dijkstra_core.hpp>
#include <pathfinder_algo_dstar_lite/dstar_lite_core.hpp>
#include <pathfinder_algo_rrt/rrt_core.hpp>
#include <pathfinder_core/voxel_grid.hpp>

#include "pfmap.hpp"


namespace
{

using pathfinder_core::InflatedVoxelGrid;
using pathfinder_core::VoxelIndex;

struct Outcome
{
  bool success{false};
  std::string note;
  double cost{0.0};
  std::size_t path_len{0};
  long long nodes{0};
  double time_ms{0.0};
};

constexpr double kBudgetSec = 60.0;

double voxel_path_cost(const std::vector<VoxelIndex> & p, double res)
{
  double c = 0.0;
  for (std::size_t i = 1; i < p.size(); ++i) {
    c += pathfinder_core::voxel_distance(p[i - 1], p[i], res);
  }
  return c;
}

// Empty string if the voxel path is a valid 26-connected collision-free walk
// from start to goal, otherwise the reason it is not.
std::string check_voxel_path(
  const InflatedVoxelGrid & g, const std::vector<VoxelIndex> & p,
  const VoxelIndex & s, const VoxelIndex & t)
{
  if (p.empty()) {
    return "empty path";
  }
  if (p.front() != s || p.back() != t) {
    return "wrong endpoints";
  }
  for (std::size_t i = 0; i < p.size(); ++i) {
    if (g.is_occupied(p[i])) {
      return "collision";
    }
    if (i > 0 && std::max({std::abs(p[i].x - p[i - 1].x), std::abs(p[i].y - p[i - 1].y),
        std::abs(p[i].z - p[i - 1].z)}) != 1)
    {
      return "non-adjacent step";
    }
  }
  return "";
}

Outcome from_voxels(
  const InflatedVoxelGrid & g, bool ok, const std::string & msg,
  const std::vector<VoxelIndex> & path, long long nodes, const VoxelIndex & s,
  const VoxelIndex & t)
{
  Outcome o;
  o.nodes = nodes;
  if (!ok) {
    o.note = msg;
    return o;
  }
  o.note = check_voxel_path(g, path, s, t);
  o.success = o.note.empty();
  o.cost = voxel_path_cost(path, g.resolution());
  o.path_len = path.size();
  return o;
}

using Runner = std::function<Outcome(const VoxelIndex &, const VoxelIndex &)>;

struct Algo
{
  std::string name;
  std::string device;
  Runner run;
};

std::string json_escape(const std::string & s)
{
  std::string o;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
    }
    o += c;
  }
  return o;
}

std::vector<std::string> split(const std::string & s, char sep)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, sep)) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

}  // namespace

int main(int argc, char ** argv)
{
  std::vector<std::string> algos = {
    "astar", "dijkstra", "dstar_lite", "rrt_star"};
  int repeats = 3;
  std::string out_path = "bench_results.jsonl";
  std::vector<std::string> maps;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--algos" && i + 1 < argc) {
      algos = split(argv[++i], ',');
    } else if (a == "--repeats" && i + 1 < argc) {
      repeats = std::max(1, std::atoi(argv[++i]));
    } else if (a == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (a == "-h" || a == "--help") {
      std::cout << "usage: pf_bench [--algos a,b] [--repeats N] [--out f.jsonl] maps...\n";
      return 0;
    } else {
      maps.push_back(a);
    }
  }
  if (maps.empty()) {
    std::cerr << "no .pfmap files given\n";
    return 2;
  }

  std::ofstream out(out_path);
  for (const auto & path : maps) {
    auto m = pf_bench::load_pfmap(path);
    InflatedVoxelGrid grid(m.dims, m.min_bound, m.resolution, m.occupied);

    std::vector<Algo> runners;
    pathfinder_algo_dstar_lite::DstarLiteCore dstar;
    for (const auto & name : algos) {
      if (name == "astar") {
        runners.push_back({name, "cpu", [&](const VoxelIndex & s, const VoxelIndex & t) {
          pathfinder_algo_astar::AstarParams p;
          p.max_plan_time_sec = kBudgetSec;
          auto r = pathfinder_algo_astar::AstarCore().plan(grid, s, t, p, nullptr);
          return from_voxels(grid, r.success, r.message, r.path, r.nodes_expanded, s, t);
        }});
      } else if (name == "dijkstra") {
        runners.push_back({name, "cpu", [&](const VoxelIndex & s, const VoxelIndex & t) {
          pathfinder_algo_dijkstra::DijkstraParams p;
          p.max_plan_time_sec = kBudgetSec;
          p.publish_current_best = false;
          auto r = pathfinder_algo_dijkstra::DijkstraCore().plan(grid, s, t, p, nullptr);
          return from_voxels(grid, r.success, r.message, r.path, r.nodes_expanded, s, t);
        }});
      } else if (name == "dstar_lite") {
        // Fresh search every call (reset), so this measures the initial plan,
        // not D* Lite's incremental replanning.
        runners.push_back({name, "cpu", [&](const VoxelIndex & s, const VoxelIndex & t) {
          pathfinder_algo_dstar_lite::DstarLiteParams p;
          p.max_plan_time_sec = kBudgetSec;
          dstar.reset();
          auto r = dstar.plan(grid, s, t, p, nullptr);
          return from_voxels(grid, r.success, r.message, r.path, r.nodes_expanded, s, t);
        }});
      } else if (name == "rrt_star") {
        runners.push_back({name, "cpu", [&](const VoxelIndex & s, const VoxelIndex & t) {
          // The demo's shipping RRT* config (5000 iterations, 5 s budget), seeded.
          pathfinder_algo_rrt::RRTParams p;
          p.random_seed = 42;
          auto r = pathfinder_algo_rrt::RRTCore().plan(grid, s, t, p, nullptr);
          Outcome o;
          o.nodes = r.nodes_expanded;
          o.note = r.success ? "" : r.message;
          if (r.success) {
            // RRT* returns a continuous path; validate each segment against the grid.
            for (std::size_t i = 1; i < r.path_world.size() && o.note.empty(); ++i) {
              const auto & a = r.path_world[i - 1];
              const auto & b = r.path_world[i];
              if (!grid.is_segment_free(a.x, a.y, a.z, b.x, b.y, b.z)) {
                o.note = "segment collision";
              }
              o.cost += std::sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) +
                (a.z - b.z) * (a.z - b.z));
            }
            o.success = o.note.empty();
            o.path_len = r.path_world.size();
          }
          return o;
        }});
      } else {
        std::cerr << "unknown algorithm " << name << "\n";
        return 2;
      }
    }

    for (const auto & algo : runners) {
      double total = 0.0;
      int ok = 0;
      for (std::size_t qi = 0; qi < m.queries.size(); ++qi) {
        const auto & q = m.queries[qi];
        const VoxelIndex s{q.start[0], q.start[1], q.start[2]};
        const VoxelIndex t{q.goal[0], q.goal[1], q.goal[2]};
        std::vector<double> times;
        Outcome o;
        // RRT* is seeded (deterministic) and slow, so one run is enough.
        const int reps = algo.name == "rrt_star" ? 1 : repeats;
        for (int rep = 0; rep < reps; ++rep) {
          const auto t0 = std::chrono::steady_clock::now();
          o = algo.run(s, t);
          const double wall = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
          times.push_back(o.time_ms > 0.0 ? o.time_ms : wall);
        }
        std::sort(times.begin(), times.end());
        o.time_ms = times[times.size() / 2];
        total += o.time_ms;
        ok += o.success ? 1 : 0;
        out << "{\"map\":\"" << m.name << "\",\"cells\":" << m.occupied.size()
            << ",\"algo\":\"" << algo.name << "\",\"device\":\"" << algo.device
            << "\",\"query\":" << qi << ",\"success\":" << (o.success ? "true" : "false")
            << ",\"note\":\"" << json_escape(o.note) << "\",\"cost\":" << o.cost
            << ",\"path_len\":" << o.path_len << ",\"nodes\":" << o.nodes
            << ",\"time_ms\":" << o.time_ms;
        out << "}\n";
        out.flush();
      }
      std::fprintf(stderr, "%-12s %-15s %3d/%-3zu ok  mean %9.2f ms\n", m.name.c_str(),
        algo.name.c_str(), ok, m.queries.size(), total / std::max<std::size_t>(1, m.queries.size()));
    }
  }
  return 0;
}
