// Convert an OctoMap .bt file into an inflated .pfmap grid using the exact
// InflatedVoxelGrid code the ROS planners use.
//   pf_export_bt <in.bt> <out.pfmap> [robot_radius=0.25]

#include <cstdio>
#include <cstdlib>
#include <string>

#include <octomap/OcTree.h>

#include <pathfinder_core/voxel_grid.hpp>

#include "pfmap.hpp"

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <in.bt> <out.pfmap> [robot_radius=0.25]\n", argv[0]);
    return 2;
  }
  const double radius = argc > 3 ? std::atof(argv[3]) : 0.25;
  octomap::OcTree tree(argv[1]);
  const pathfinder_core::InflatedVoxelGrid grid(tree, radius);

  pf_bench::PfMap m;
  m.dims = grid.dims();
  m.min_bound = grid.min_bound();
  m.resolution = grid.resolution();
  m.occupied.resize(grid.cell_count());
  for (std::size_t i = 0; i < grid.cell_count(); ++i) {
    m.occupied[i] = grid.is_occupied(grid.from_linear_index(i)) ? 1 : 0;
  }
  pf_bench::save_pfmap(argv[2], m);
  std::printf("%s: %dx%dx%d @ %.3f m, min (%.2f, %.2f, %.2f)\n", argv[2],
    m.dims[0], m.dims[1], m.dims[2], m.resolution,
    m.min_bound[0], m.min_bound[1], m.min_bound[2]);
  return 0;
}
