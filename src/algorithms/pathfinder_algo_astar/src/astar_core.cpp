#include "pathfinder_algo_astar/astar_core.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <queue>
#include <thread>
#include <vector>

#include <pathfinder_core/marker_decimation.hpp>
#include <pathfinder_core/grid_utils.hpp>
#include <pathfinder_core/voxel_grid.hpp>

namespace pathfinder_algo_astar
{

namespace
{

using pathfinder_core::InflatedVoxelGrid;
using pathfinder_core::VoxelIndex;
using pathfinder_core::voxel_distance;
using pathfinder_core::kMaxVizVoxels;
using pathfinder_core::sample_flag_indices;
using pathfinder_core::reconstruct_path_from_parents;

struct OpenEntry
{
  double f = 0.0;
  double h = 0.0;
  std::size_t flat = 0;

  bool operator>(const OpenEntry & other) const
  {
    if (f != other.f) {
      return f > other.f;
    }
    return h > other.h;
  }
};

using OpenQueue = std::priority_queue<
  OpenEntry, std::vector<OpenEntry>, std::greater<OpenEntry>>;

struct SearchBuffers
{
  std::vector<double> g_score;
  std::vector<int32_t> came_from;
  std::vector<uint8_t> closed;
  std::vector<uint8_t> in_open;

  void reset(std::size_t n)
  {
    g_score.assign(n, std::numeric_limits<double>::infinity());
    came_from.assign(n, -1);
    closed.assign(n, 0);
    in_open.assign(n, 0);
  }
};

}  // namespace

AstarResult AstarCore::plan(
  const InflatedVoxelGrid & grid,
  const VoxelIndex & start,
  const VoxelIndex & goal,
  const AstarParams & params,
  FeedbackCallback feedback)
{
  AstarResult result;
  const auto start_wall = std::chrono::steady_clock::now();

  const double res = grid.resolution();
  const int feedback_every_nodes = std::max(1, params.feedback_every_nodes);
  const double feedback_every_seconds = params.feedback_every_seconds;
  const double max_plan_time_sec = params.max_plan_time_sec;

  const std::size_t start_flat = grid.linear_index(start);
  const std::size_t goal_flat = grid.linear_index(goal);

  SearchBuffers buf;
  buf.reset(grid.cell_count());
  buf.g_score[start_flat] = 0.0;

  OpenQueue open;
  const double h_start = voxel_distance(start, goal, res);
  open.push({h_start, h_start, start_flat});
  buf.in_open[start_flat] = 1;

  std::size_t best_open_flat = start_flat;
  double best_open_f = h_start;
  int nodes_expanded = 0;
  int nodes_since_feedback = 0;
  auto last_feedback_time = start_wall;
  bool reached = false;
  bool timed_out = false;

  auto emit_feedback = [&](bool include_best_path) {
    if (!feedback) {
      return;
    }
    AstarFeedback fb;
    fb.nodes_expanded = nodes_expanded;

    {
      OpenQueue copy = open;
      while (!copy.empty() && fb.sampled_open_flat.size() < kMaxVizVoxels) {
        fb.sampled_open_flat.push_back(copy.top().flat);
        copy.pop();
      }
    }

    fb.sampled_closed_flat = sample_flag_indices(buf.closed, kMaxVizVoxels);

    if (include_best_path) {
      fb.best_partial_path = reconstruct_path_from_parents(
        buf.came_from, grid, best_open_flat);
      const auto best_v = grid.from_linear_index(best_open_flat);
      fb.best_cost_so_far = buf.g_score[best_open_flat] +
        voxel_distance(best_v, goal, res);
    }

    feedback(fb);
  };

  double slept_sec = 0.0;  // viz-delay sleeps must not count against the compute budget
  while (!open.empty()) {
    const auto now_wall = std::chrono::steady_clock::now();
    const double elapsed_sec =
      std::chrono::duration<double>(now_wall - start_wall).count();
    if (elapsed_sec - slept_sec > max_plan_time_sec) {
      timed_out = true;
      break;
    }

    OpenEntry cur = open.top();
    open.pop();

    if (buf.closed[cur.flat]) {
      continue;
    }
    buf.closed[cur.flat] = 1;
    buf.in_open[cur.flat] = 0;
    ++nodes_expanded;
    ++nodes_since_feedback;

    if (cur.flat == goal_flat) {
      reached = true;
      break;
    }

    if (cur.h < best_open_f) {
      best_open_f = cur.h;
      best_open_flat = cur.flat;
    }

    const VoxelIndex cv = grid.from_linear_index(cur.flat);
    for (const auto & nb : grid.neighbors_26(cv)) {
      const std::size_t nflat = grid.linear_index(nb);
      if (buf.closed[nflat]) {
        continue;
      }
      if (grid.is_occupied(nb)) {
        continue;
      }
      const double step = voxel_distance(cv, nb, res);
      const double tentative_g = buf.g_score[cur.flat] + step;
      if (tentative_g >= buf.g_score[nflat]) {
        continue;
      }
      buf.g_score[nflat] = tentative_g;
      buf.came_from[nflat] = static_cast<int32_t>(cur.flat);
      const double h = voxel_distance(nb, goal, res);
      open.push({tentative_g + h, h, nflat});
      buf.in_open[nflat] = 1;
    }

    const double since_fb_sec =
      std::chrono::duration<double>(now_wall - last_feedback_time).count();
    if (nodes_since_feedback >= feedback_every_nodes ||
        since_fb_sec >= feedback_every_seconds)
    {
      emit_feedback(true);
      if (params.viz_delay_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(params.viz_delay_ms));
        slept_sec += params.viz_delay_ms / 1000.0;
      }
      nodes_since_feedback = 0;
      last_feedback_time = now_wall;
    }
  }

  const auto end_wall = std::chrono::steady_clock::now();
  result.plan_duration_ms =
    std::chrono::duration<double, std::milli>(end_wall - start_wall).count();
  result.nodes_expanded = nodes_expanded;

  if (timed_out) {
    result.success = false;
    result.message = "time budget exceeded";
    return result;
  }
  if (!reached) {
    result.success = false;
    result.message = "no path found";
    return result;
  }

  result.path = reconstruct_path_from_parents(buf.came_from, grid, goal_flat);
  result.success = true;
  result.message = "ok";
  return result;
}

}  // namespace pathfinder_algo_astar
