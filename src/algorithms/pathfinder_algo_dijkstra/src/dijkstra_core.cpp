#include "pathfinder_algo_dijkstra/dijkstra_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <thread>
#include <vector>

#include <pathfinder_core/marker_decimation.hpp>
#include <pathfinder_core/grid_utils.hpp>
#include <pathfinder_core/voxel_grid.hpp>

namespace pathfinder_algo_dijkstra
{

DijkstraResult DijkstraCore::plan(
  const pathfinder_core::InflatedVoxelGrid & grid,
  const pathfinder_core::VoxelIndex & start,
  const pathfinder_core::VoxelIndex & goal,
  const DijkstraParams & params,
  FeedbackCallback feedback)
{
  const auto clock_start = std::chrono::steady_clock::now();

  DijkstraResult out{};
  out.success = false;
  out.nodes_expanded = 0;

  auto finalize_duration = [&]() {
    const auto clock_end = std::chrono::steady_clock::now();
    out.plan_duration_ms = std::chrono::duration<double, std::milli>(
      clock_end - clock_start).count();
  };

  if (!grid.in_bounds(start)) {
    out.message = "start is outside map bounds";
    finalize_duration();
    return out;
  }
  if (!grid.in_bounds(goal)) {
    out.message = "goal is outside map bounds";
    finalize_duration();
    return out;
  }
  if (grid.is_occupied(start)) {
    out.message = "start voxel is occupied (after inflation)";
    finalize_duration();
    return out;
  }
  if (grid.is_occupied(goal)) {
    out.message = "goal voxel is occupied (after inflation)";
    finalize_duration();
    return out;
  }

  const double res = grid.resolution();
  std::array<double, 26> step_costs{};
  for (std::size_t i = 0; i < pathfinder_core::kNeighborOffsets26.size(); ++i) {
    const auto & d = pathfinder_core::kNeighborOffsets26[i];
    step_costs[i] = res * std::sqrt(
      static_cast<double>(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));
  }

  const std::size_t n_cells = grid.cell_count();
  std::vector<double> g(n_cells, std::numeric_limits<double>::infinity());
  std::vector<std::int32_t> parent(n_cells, -1);
  std::vector<std::uint8_t> closed(n_cells, 0);
  std::vector<std::uint8_t> in_open(n_cells, 0);

  struct PqEntry
  {
    double cost;
    std::int32_t idx;
  };
  struct PqCmp
  {
    bool operator()(const PqEntry & a, const PqEntry & b) const
    {
      return a.cost > b.cost;
    }
  };
  std::priority_queue<PqEntry, std::vector<PqEntry>, PqCmp> open;

  const auto dims = grid.dims();
  const int dim_x = dims[0];
  const int dim_y = dims[1];
  const int dim_z = dims[2];

  const std::int32_t start_lin = static_cast<std::int32_t>(grid.linear_index(start));
  const std::int32_t goal_lin = static_cast<std::int32_t>(grid.linear_index(goal));
  g[start_lin] = 0.0;
  open.push({0.0, start_lin});
  in_open[start_lin] = 1;

  auto deadline = std::chrono::steady_clock::now() +
    std::chrono::duration<double>(params.max_plan_time_sec);
  auto last_feedback = std::chrono::steady_clock::now();
  std::int32_t expanded = 0;
  std::int32_t nodes_since_feedback = 0;

  while (!open.empty()) {
    if (std::chrono::steady_clock::now() > deadline) {
      out.message = "time budget exceeded";
      out.nodes_expanded = expanded;
      finalize_duration();
      return out;
    }

    PqEntry top = open.top();
    open.pop();
    if (closed[top.idx]) {
      continue;
    }
    if (top.cost > g[top.idx] + 1e-9) {
      continue;
    }
    closed[top.idx] = 1;
    in_open[top.idx] = 0;
    ++expanded;
    ++nodes_since_feedback;

    if (top.idx == goal_lin) {
      out.success = true;
      out.message = "ok";
      out.nodes_expanded = expanded;
      out.path = pathfinder_core::reconstruct_path_from_parents(
        parent, grid, static_cast<std::size_t>(top.idx));
      finalize_duration();
      return out;
    }

    const pathfinder_core::VoxelIndex cur = grid.from_linear_index(
      static_cast<std::size_t>(top.idx));
    for (std::size_t k = 0; k < pathfinder_core::kNeighborOffsets26.size(); ++k) {
      const auto & d = pathfinder_core::kNeighborOffsets26[k];
      const int nx = cur.x + d[0];
      const int ny = cur.y + d[1];
      const int nz = cur.z + d[2];
      if (nx < 0 || ny < 0 || nz < 0 || nx >= dim_x || ny >= dim_y || nz >= dim_z) {
        continue;
      }
      const std::size_t nb_lin = grid.linear_index({nx, ny, nz});
      if (grid.is_occupied({nx, ny, nz}) || closed[nb_lin]) {
        continue;
      }
      const double tentative = top.cost + step_costs[k];
      if (tentative < g[nb_lin] - 1e-9) {
        g[nb_lin] = tentative;
        parent[nb_lin] = top.idx;
        open.push({tentative, static_cast<std::int32_t>(nb_lin)});
        in_open[nb_lin] = 1;
      }
    }

    const bool by_count = params.feedback_every_nodes > 0 &&
      (nodes_since_feedback >= params.feedback_every_nodes);
    bool by_time = false;
    if (params.feedback_every_seconds > 0.0 && (expanded & 511) == 0) {
      const auto now = std::chrono::steady_clock::now();
      by_time = std::chrono::duration<double>(now - last_feedback).count() >=
        params.feedback_every_seconds;
    }
    if (feedback && (by_count || by_time)) {
      last_feedback = std::chrono::steady_clock::now();
      nodes_since_feedback = 0;
      DijkstraFeedback fb{};
      fb.nodes_expanded = expanded;
      fb.best_cost_so_far = top.cost;
      fb.best_partial_path = pathfinder_core::reconstruct_path_from_parents(
        parent, grid, static_cast<std::size_t>(top.idx));
      fb.sampled_closed_flat = pathfinder_core::sample_flag_indices(
        closed, pathfinder_core::kMaxVizVoxels);
      fb.sampled_open_flat = pathfinder_core::sample_flag_indices(
        in_open, pathfinder_core::kMaxVizVoxels);
      feedback(fb);
      if (params.viz_delay_ms > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(params.viz_delay_ms));
        // exclude the intentional viz pause from the compute budget
        deadline += std::chrono::milliseconds(params.viz_delay_ms);
      }
    }
  }

  out.message = "no path found";
  out.nodes_expanded = expanded;
  finalize_duration();
  return out;
}

}  // namespace pathfinder_algo_dijkstra
