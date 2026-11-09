#include "gpu_planners.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

namespace pf_gpu
{

namespace
{

#define PF_CUDA_CHECK(expr)                                                   \
  do {                                                                        \
    const cudaError_t err_ = (expr);                                          \
    if (err_ != cudaSuccess) {                                                \
      throw std::runtime_error(std::string(#expr) + ": " +                    \
        cudaGetErrorString(err_));                                            \
    }                                                                         \
  } while (0)

constexpr int kTile = 8;                // interior tile edge for the wavefront
constexpr int kHalo = kTile + 2;
constexpr int kLocalIters = 3 * kTile;  // enough to cross a tile diagonally
constexpr int kInfBits = 0x7f800000;    // +inf; non-negative floats order like ints
constexpr int kMaxPath = 1 << 20;

// 26-neighbour offsets and their step lengths in voxels (1, sqrt2, sqrt3).
__constant__ int c_off[26][3];
__constant__ float c_len[26];

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

__device__ __forceinline__ bool inside(int x, int y, int z, int nx, int ny, int nz)
{
  return x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz;
}

__global__ void fill_inf(int * dist, long long n)
{
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i < n) {
    dist[i] = kInfBits;
  }
}

// One tile per block, one voxel per thread. The tile plus a one-voxel halo is
// staged in shared memory and relaxed in place until it stops changing
// (or kLocalIters), then improved interior values are written back. Halo values
// may be stale within a launch; relaxation is monotone, so later launches fix it.
__global__ void relax_tiles(
  const std::uint8_t * occ, float * dist, int nx, int ny, int nz, float res, int * changed)
{
  __shared__ float s[kHalo][kHalo][kHalo];
  const int bx = blockIdx.x * kTile, by = blockIdx.y * kTile, bz = blockIdx.z * kTile;
  const int tid = threadIdx.x + kTile * (threadIdx.y + kTile * threadIdx.z);
  for (int i = tid; i < kHalo * kHalo * kHalo; i += kTile * kTile * kTile) {
    const int lx = i % kHalo, ly = (i / kHalo) % kHalo, lz = i / (kHalo * kHalo);
    const int gx = bx + lx - 1, gy = by + ly - 1, gz = bz + lz - 1;
    s[lz][ly][lx] = inside(gx, gy, gz, nx, ny, nz) ?
      dist[(static_cast<long long>(gz) * ny + gy) * nx + gx] : __int_as_float(kInfBits);
  }
  __syncthreads();

  const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  const int gx = bx + tx, gy = by + ty, gz = bz + tz;
  const long long g = (static_cast<long long>(gz) * ny + gy) * nx + gx;
  const bool active = inside(gx, gy, gz, nx, ny, nz) && occ[g] == 0;
  float mine = s[tz + 1][ty + 1][tx + 1];
  bool improved = false;
  for (int it = 0; it < kLocalIters; ++it) {
    float best = mine;
    if (active) {
      #pragma unroll
      for (int k = 0; k < 26; ++k) {
        best = fminf(best,
          s[tz + 1 + c_off[k][2]][ty + 1 + c_off[k][1]][tx + 1 + c_off[k][0]] + c_len[k] * res);
      }
    }
    const int upd = best < mine;
    __syncthreads();
    if (upd) {
      s[tz + 1][ty + 1][tx + 1] = best;
      mine = best;
      improved = true;
    }
    if (!__syncthreads_or(upd)) {
      break;
    }
  }
  if (improved) {
    dist[g] = mine;
    *changed = 1;
  }
}

// Sparse A*-pruned relaxation from the goal towards `start`. Each frontier voxel
// relaxes its neighbours with atomicMin on the float bits; a neighbour that
// improves joins the next frontier once (stamp de-dup). Voxels with
// g + h(start) >= cost(start) cannot improve the answer and are skipped.
__global__ void expand_frontier(
  const std::uint8_t * occ, int * dist, int nx, int ny, int nz, float res,
  const int * frontier, int fsize, int * next, int * next_size,
  int * stamp_of, int stamp, int sx, int sy, int sz, long long start_flat)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= fsize) {
    return;
  }
  const int u = frontier[i];
  const float du = __int_as_float(dist[u]);
  const float bound = __int_as_float(dist[start_flat]);
  const int ux = u % nx, uy = (u / nx) % ny, uz = u / (nx * ny);
  const float hu = 0.999f * res * sqrtf(
    float((ux - sx) * (ux - sx) + (uy - sy) * (uy - sy) + (uz - sz) * (uz - sz)));
  if (du + hu >= bound) {
    return;
  }
  #pragma unroll 2
  for (int k = 0; k < 26; ++k) {
    const int x = ux + c_off[k][0], y = uy + c_off[k][1], z = uz + c_off[k][2];
    if (!inside(x, y, z, nx, ny, nz)) {
      continue;
    }
    const int v = (z * ny + y) * nx + x;
    if (occ[v]) {
      continue;
    }
    const float nd = du + c_len[k] * res;
    const float hv = 0.999f * res * sqrtf(
      float((x - sx) * (x - sx) + (y - sy) * (y - sy) + (z - sz) * (z - sz)));
    if (nd + hv >= bound) {
      continue;
    }
    const int nd_bits = __float_as_int(nd);
    if (nd_bits < atomicMin(&dist[v], nd_bits)) {
      if (atomicExch(&stamp_of[v], stamp) != stamp) {
        next[atomicAdd(next_size, 1)] = v;
      }
    }
  }
}

// Single-thread descent of the cost field from start to goal. len < 0 means the
// walk got stuck (should not happen on a converged field).
__global__ void extract_path(
  const std::uint8_t * occ, const float * dist, int nx, int ny, int nz, float res,
  int sx, int sy, int sz, int3 * path, int * len, float * cost)
{
  int x = sx, y = sy, z = sz;
  long long cur = (static_cast<long long>(z) * ny + y) * nx + x;
  if (!(dist[cur] < __int_as_float(kInfBits))) {
    *len = 0;
    *cost = 0.0f;
    return;
  }
  int n = 0;
  float total = 0.0f;
  path[n++] = make_int3(x, y, z);
  while (dist[cur] > 0.0f && n < kMaxPath) {
    float best = __int_as_float(kInfBits);
    int bk = -1;
    for (int k = 0; k < 26; ++k) {
      const int X = x + c_off[k][0], Y = y + c_off[k][1], Z = z + c_off[k][2];
      if (!inside(X, Y, Z, nx, ny, nz)) {
        continue;
      }
      const long long v = (static_cast<long long>(Z) * ny + Y) * nx + X;
      if (occ[v] || !(dist[v] < dist[cur])) {
        continue;
      }
      const float val = dist[v] + c_len[k] * res;
      if (val < best) {
        best = val;
        bk = k;
      }
    }
    if (bk < 0) {
      break;
    }
    x += c_off[bk][0];
    y += c_off[bk][1];
    z += c_off[bk][2];
    cur = (static_cast<long long>(z) * ny + y) * nx + x;
    total += c_len[bk] * res;
    path[n++] = make_int3(x, y, z);
  }
  *len = dist[cur] == 0.0f ? n : -n;
  *cost = total;
}

}  // namespace

struct GpuGrid::Impl
{
  int nx, ny, nz;
  long long n;
  float res;
  std::uint8_t * occ{nullptr};
  int * dist{nullptr};
  int * front_a{nullptr};
  int * front_b{nullptr};
  int * stamp_of{nullptr};
  int * counter{nullptr};  // [0] changed / next_size, [1] path len
  float * path_cost{nullptr};
  int3 * path{nullptr};
  int * pinned{nullptr};   // host-pinned mirror of counter
  int stamp{0};

  void reset_field(long long goal_flat)
  {
    const int threads = 256;
    fill_inf<<<static_cast<unsigned>((n + threads - 1) / threads), threads>>>(dist, n);
    const float zero = 0.0f;
    PF_CUDA_CHECK(cudaMemcpy(dist + goal_flat, &zero, sizeof(float), cudaMemcpyHostToDevice));
  }

  GpuResult finish(std::array<int, 3> start, Clock::time_point t0, GpuResult r)
  {
    extract_path<<<1, 1>>>(occ, reinterpret_cast<float *>(dist), nx, ny, nz, res,
      start[0], start[1], start[2], path, counter + 1, path_cost);
    int len = 0;
    PF_CUDA_CHECK(cudaMemcpy(&len, counter + 1, sizeof(int), cudaMemcpyDeviceToHost));
    float cost = 0.0f;
    PF_CUDA_CHECK(cudaMemcpy(&cost, path_cost, sizeof(float), cudaMemcpyDeviceToHost));
    if (len > 0) {
      std::vector<int3> buf(static_cast<std::size_t>(len));
      PF_CUDA_CHECK(cudaMemcpy(buf.data(), path, sizeof(int3) * buf.size(), cudaMemcpyDeviceToHost));
      r.path.reserve(buf.size());
      for (const auto & p : buf) {
        r.path.push_back({p.x, p.y, p.z});
      }
      r.success = true;
      r.cost = cost;
    }
    r.time_ms = ms_since(t0);
    return r;
  }
};

bool available()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::string device_name()
{
  cudaDeviceProp p{};
  if (!available() || cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
    return "none";
  }
  return p.name;
}

GpuGrid::GpuGrid(std::array<int, 3> dims, const std::uint8_t * occupied, double resolution)
: impl_(std::make_unique<Impl>())
{
  const auto t0 = Clock::now();
  auto & m = *impl_;
  m.nx = dims[0];
  m.ny = dims[1];
  m.nz = dims[2];
  m.n = static_cast<long long>(m.nx) * m.ny * m.nz;
  m.res = static_cast<float>(resolution);
  if (m.n >= (1LL << 31)) {
    throw std::runtime_error("grid too large for 32-bit voxel indices");
  }

  int off[26][3];
  float len[26];
  int k = 0;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) {
          continue;
        }
        off[k][0] = dx;
        off[k][1] = dy;
        off[k][2] = dz;
        len[k] = std::sqrt(static_cast<float>(dx * dx + dy * dy + dz * dz));
        ++k;
      }
    }
  }
  PF_CUDA_CHECK(cudaMemcpyToSymbol(c_off, off, sizeof(off)));
  PF_CUDA_CHECK(cudaMemcpyToSymbol(c_len, len, sizeof(len)));

  const auto n = static_cast<std::size_t>(m.n);
  PF_CUDA_CHECK(cudaMalloc(&m.occ, n));
  PF_CUDA_CHECK(cudaMalloc(&m.dist, n * sizeof(int)));
  PF_CUDA_CHECK(cudaMalloc(&m.front_a, n * sizeof(int)));
  PF_CUDA_CHECK(cudaMalloc(&m.front_b, n * sizeof(int)));
  PF_CUDA_CHECK(cudaMalloc(&m.stamp_of, n * sizeof(int)));
  PF_CUDA_CHECK(cudaMalloc(&m.counter, 2 * sizeof(int)));
  PF_CUDA_CHECK(cudaMalloc(&m.path_cost, sizeof(float)));
  PF_CUDA_CHECK(cudaMalloc(&m.path, kMaxPath * sizeof(int3)));
  PF_CUDA_CHECK(cudaMallocHost(&m.pinned, sizeof(int)));
  PF_CUDA_CHECK(cudaMemcpy(m.occ, occupied, n, cudaMemcpyHostToDevice));
  PF_CUDA_CHECK(cudaMemset(m.stamp_of, 0, n * sizeof(int)));
  PF_CUDA_CHECK(cudaDeviceSynchronize());
  upload_ms_ = ms_since(t0);
}

GpuGrid::~GpuGrid()
{
  auto & m = *impl_;
  cudaFree(m.occ);
  cudaFree(m.dist);
  cudaFree(m.front_a);
  cudaFree(m.front_b);
  cudaFree(m.stamp_of);
  cudaFree(m.counter);
  cudaFree(m.path_cost);
  cudaFree(m.path);
  cudaFreeHost(m.pinned);
}

GpuResult GpuGrid::wavefront(std::array<int, 3> start, std::array<int, 3> goal)
{
  auto & m = *impl_;
  const auto t0 = Clock::now();
  GpuResult r;
  const long long goal_flat = (static_cast<long long>(goal[2]) * m.ny + goal[1]) * m.nx + goal[0];
  m.reset_field(goal_flat);
  const dim3 block(kTile, kTile, kTile);
  const dim3 grid((m.nx + kTile - 1) / kTile, (m.ny + kTile - 1) / kTile, (m.nz + kTile - 1) / kTile);
  const long long tiles = static_cast<long long>(grid.x) * grid.y * grid.z;
  for (;;) {
    PF_CUDA_CHECK(cudaMemsetAsync(m.counter, 0, sizeof(int)));
    relax_tiles<<<grid, block>>>(m.occ, reinterpret_cast<float *>(m.dist),
      m.nx, m.ny, m.nz, m.res, m.counter);
    PF_CUDA_CHECK(cudaMemcpy(m.pinned, m.counter, sizeof(int), cudaMemcpyDeviceToHost));
    ++r.iterations;
    r.work += tiles;
    if (*m.pinned == 0) {
      break;
    }
  }
  PF_CUDA_CHECK(cudaGetLastError());
  return m.finish(start, t0, r);
}

GpuResult GpuGrid::frontier(std::array<int, 3> start, std::array<int, 3> goal)
{
  auto & m = *impl_;
  const auto t0 = Clock::now();
  GpuResult r;
  const long long goal_flat = (static_cast<long long>(goal[2]) * m.ny + goal[1]) * m.nx + goal[0];
  const long long start_flat = (static_cast<long long>(start[2]) * m.ny + start[1]) * m.nx + start[0];
  m.reset_field(goal_flat);
  const int g = static_cast<int>(goal_flat);
  PF_CUDA_CHECK(cudaMemcpy(m.front_a, &g, sizeof(int), cudaMemcpyHostToDevice));
  int fsize = 1;
  const int threads = 256;
  while (fsize > 0) {
    ++m.stamp;
    PF_CUDA_CHECK(cudaMemsetAsync(m.counter, 0, sizeof(int)));
    expand_frontier<<<(fsize + threads - 1) / threads, threads>>>(
      m.occ, m.dist, m.nx, m.ny, m.nz, m.res, m.front_a, fsize, m.front_b, m.counter,
      m.stamp_of, m.stamp, start[0], start[1], start[2], start_flat);
    PF_CUDA_CHECK(cudaMemcpy(m.pinned, m.counter, sizeof(int), cudaMemcpyDeviceToHost));
    r.work += fsize;
    ++r.iterations;
    fsize = *m.pinned;
    std::swap(m.front_a, m.front_b);
  }
  PF_CUDA_CHECK(cudaGetLastError());
  return m.finish(start, t0, r);
}

}  // namespace pf_gpu
