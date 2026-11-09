#pragma once

#include <utility>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pathfinder_msgs/action/request_path.hpp>

#include "pathfinder_core/grid_utils.hpp"
#include "pathfinder_core/voxel_grid.hpp"

namespace pathfinder_core
{

// Convert a RequestPath goal's start/goal poses to VoxelIndex coordinates.
inline std::pair<VoxelIndex, VoxelIndex> voxels_from_goal(
  const InflatedVoxelGrid & grid,
  const pathfinder_msgs::action::RequestPath::Goal & goal)
{
  const VoxelIndex start = grid.world_to_voxel(
    goal.start.pose.position.x,
    goal.start.pose.position.y,
    goal.start.pose.position.z);
  const VoxelIndex end = grid.world_to_voxel(
    goal.goal.pose.position.x,
    goal.goal.pose.position.y,
    goal.goal.pose.position.z);
  return {start, end};
}

}  // namespace pathfinder_core
