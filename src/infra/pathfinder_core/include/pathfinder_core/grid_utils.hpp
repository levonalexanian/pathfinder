#pragma once

// ROS-free grid helpers shared by the planner cores. Kept separate from
// planner_utils.hpp (which needs ROS message types) so the cores can be built
// and benchmarked without a ROS workspace.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "pathfinder_core/voxel_grid.hpp"

namespace pathfinder_core
{

// Nudge both start and goal to a nearby free voxel. nudge_radius is computed
// the same way as every caller: max(4, ceil(0.5 / max(res, 1e-3))).
// Returns nullopt for either component if no free voxel exists within the radius.
inline std::optional<std::pair<VoxelIndex, VoxelIndex>> nudge_start_and_goal(
  const InflatedVoxelGrid & grid,
  const VoxelIndex & start,
  const VoxelIndex & goal)
{
  const double res = grid.resolution();
  const int nudge_radius = std::max(
    4, static_cast<int>(std::ceil(0.5 / std::max(res, 1e-3))));
  const auto start_opt = grid.nudge_to_free(start, nudge_radius);
  const auto goal_opt = grid.nudge_to_free(goal, nudge_radius);
  if (!start_opt || !goal_opt) {
    return std::nullopt;
  }
  return std::make_pair(*start_opt, *goal_opt);
}

// Reconstruct a voxel path by walking the parent array from goal_flat back to
// the root (parent < 0). The cycle guard caps the walk at 100 000 steps,
// matching Dijkstra's existing guard, so a malformed parent array cannot loop.
inline std::vector<VoxelIndex> reconstruct_path_from_parents(
  const std::vector<std::int32_t> & parent,
  const InflatedVoxelGrid & grid,
  std::size_t goal_flat)
{
  std::vector<std::int32_t> chain;
  chain.push_back(static_cast<std::int32_t>(goal_flat));
  while (parent[chain.back()] >= 0) {
    chain.push_back(parent[chain.back()]);
    if (chain.size() > 100000) {
      break;
    }
  }
  std::reverse(chain.begin(), chain.end());
  std::vector<VoxelIndex> path;
  path.reserve(chain.size());
  for (auto lin : chain) {
    path.push_back(grid.from_linear_index(static_cast<std::size_t>(lin)));
  }
  return path;
}

}  // namespace pathfinder_core
