#pragma once

// CUDA planners on the dense 26-connected voxel grid. The header is plain C++
// so the benchmark can link it without including CUDA headers.
//
//  * wavefront: tiled, in-place Bellman-Ford relaxation of the full cost-to-goal
//    field. Every 8^3 tile iterates to a local fixed point in shared memory
//    before writing back, so each launch moves the wavefront up to 8 voxels.
//    Produces the whole field, i.e. answers every start for the goal.
//  * frontier: sparse, A*-pruned label-correcting search. Only voxels whose
//    cost improved are expanded on the next step, and any voxel that cannot
//    beat the current best start cost (g + h >= bound) is pruned. Optimal
//    because h (Euclidean) is consistent on this grid.
//
// Both return the same optimal cost as the CPU A* / Dijkstra cores.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pf_gpu
{

struct GpuResult
{
  bool success{false};
  std::vector<std::array<int, 3>> path;  // start -> goal, voxel coordinates
  double cost{0.0};                      // metres
  long long iterations{0};               // kernel launches (outer steps)
  long long work{0};                     // wavefront: tile launches; frontier: voxels expanded
  double time_ms{0.0};                   // wall time incl. path copy-back
};

bool available();
std::string device_name();

class GpuGrid
{
public:
  GpuGrid(std::array<int, 3> dims, const std::uint8_t * occupied, double resolution);
  ~GpuGrid();
  GpuGrid(const GpuGrid &) = delete;
  GpuGrid & operator=(const GpuGrid &) = delete;

  GpuResult wavefront(std::array<int, 3> start, std::array<int, 3> goal);
  GpuResult frontier(std::array<int, 3> start, std::array<int, 3> goal);

  double upload_ms() const { return upload_ms_; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  double upload_ms_{0.0};
};

}  // namespace pf_gpu
